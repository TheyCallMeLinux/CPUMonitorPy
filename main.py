import signal
import time
import threading
from collections import deque
from statistics import mean
from typing import Deque, Optional
import random
import psutil
from rich.console import Console
from rich.live import Live
from rich.progress import BarColumn, Progress, TextColumn
from rich.table import Table
from dataclasses import dataclass
from datetime import datetime
import csv
import os
import argparse

# === Config ===
@dataclass(frozen=True)
class Thresholds:
    cpu_warn: float = 70.0
    cpu_crit: float = 90.0
    temp_warn: float = 70.0
    temp_crit: float = 85.0

@dataclass(frozen=True)
class Settings:
    sample_interval_sec: float = 0.5
    temp_sample_interval_sec: float = 2.0
    window_duration_sec: float = 10
    progress_bar_width: int = 40
    cpu_sensor_keys: tuple = ("coretemp", "cpu_thermal", "k10temp", "acpitz")

THRESHOLDS = Thresholds()
SETTINGS = Settings()

console = Console()

# === Utility functions ===
def colorize(value: float, warn: float, crit: float, unit: str = "") -> str:
    """Return a colored string based on thresholds."""
    if value >= crit:
        color = "red"
    elif value >= warn:
        color = "yellow"
    else:
        color = "green"
    return f"[{color}]{value:.1f}{unit}[/{color}]"


# === CPU Monitor ===
class CPUMonitor(threading.Thread):
    """Monitor CPU usage & temperature with a sliding window in a separate thread."""

    def __init__(self):
        super().__init__(daemon=True)
        self.cpu_samples: Deque[float] = deque(maxlen=int(SETTINGS.window_duration_sec / SETTINGS.sample_interval_sec))
        self.temp_samples: Deque[float] = deque(maxlen=int(SETTINGS.window_duration_sec / SETTINGS.temp_sample_interval_sec))
        self.freq_samples: Deque[float] = deque(maxlen=int(SETTINGS.window_duration_sec / SETTINGS.sample_interval_sec))
        self.load_samples: Deque[tuple] = deque(maxlen=int(SETTINGS.window_duration_sec / SETTINGS.sample_interval_sec))
        self._last_temp_time = 0.0
        self.running = threading.Event()
        self.running.set()
        self._sensor_key = None
        self.start_time = datetime.now()   # record when monitoring starts
        self.end_time = None               # used for the summary report
        self.timestamps: Deque[str] = deque(maxlen=int(SETTINGS.window_duration_sec / SETTINGS.sample_interval_sec))

    def _detect_sensor_key(self):
        try:
            temps = psutil.sensors_temperatures()
        except Exception:
            return None
        for key in SETTINGS.cpu_sensor_keys:
            if key in temps:
                return key
        return None

    def run(self) -> None:
        """Main monitoring loop."""
        self._sensor_key = self._detect_sensor_key()
        while self.running.is_set():
            self.sample()
            time.sleep(SETTINGS.sample_interval_sec)

    def sample(self) -> None:
        """Record CPU usage, temperature, frequency, and load samples."""
        try:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.timestamps.append(now)

            self.cpu_samples.append(psutil.cpu_percent(interval=None))

            cpu_freq = psutil.cpu_freq()
            self.freq_samples.append(cpu_freq.current if cpu_freq else 0.0)

            self.load_samples.append(psutil.getloadavg())

            now_ts = time.time()
            if now_ts - self._last_temp_time >= SETTINGS.temp_sample_interval_sec:
                temp = self._read_temperature()
                if temp is not None:
                    self.temp_samples.append(temp)
                self._last_temp_time = now_ts
        except Exception as e:
            console.log(f"[red]Error during sampling: {e}[/red]")

    def _read_temperature(self) -> Optional[float]:
        try:
            temps = psutil.sensors_temperatures()
            if self._sensor_key and self._sensor_key in temps:
                entries = temps[self._sensor_key]
                match = next((t.current for t in entries if t.label and ("package" in t.label.lower() or "core 0" in t.label.lower())), None)
                return match or (entries[0].current if entries else None)
            return None
        except Exception:
            return None

    @staticmethod
    def avg(data: Deque[float]) -> float:
        """Return average of a deque or 0 if empty."""
        return mean(data) if data else 0.0

    def stop(self) -> None:
        """Stop monitoring."""
        self.running.clear()
        self.end_time = datetime.now()


# === UI Renderer ===
class CPUUI:
    """Handles rendering the CPU monitor data in the console."""

    def __init__(self, monitor: CPUMonitor):
        self.monitor = monitor
        self.progress = Progress(
            TextColumn("CPU Usage:"),
            BarColumn(bar_width=SETTINGS.progress_bar_width),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            transient=False,
            console=console,
            refresh_per_second=10,
        )
        self.cpu_task_id = self.progress.add_task("cpu", total=100, completed=0)
        self._cpu_info_cached = (
            psutil.cpu_count(logical=True),
            psutil.cpu_count(logical=False),
        )

    def make_renderable(self) -> Table:
        """Create a Rich renderable showing CPU usage, temperature, and system info."""
        m = self.monitor
        current_cpu = m.cpu_samples[-1] if m.cpu_samples else 0.0
        self.progress.update(self.cpu_task_id, completed=current_cpu)

        avg_cpu = m.avg(m.cpu_samples)
        peak_cpu = max(m.cpu_samples, default=0)
        cpu_load = psutil.getloadavg()

        table = Table.grid(expand=True)
        table.add_row(self.progress)
        table.add_row(
            f"Average CPU: {colorize(avg_cpu, THRESHOLDS.cpu_warn, THRESHOLDS.cpu_crit, '%')} | "
            f"Peak: {colorize(peak_cpu, THRESHOLDS.cpu_warn, THRESHOLDS.cpu_crit, '%')} | "
            f"1 minute load average: {cpu_load[0]:.2f}"
        )
        table.add_row(
            f"Logical CPU Cores: {self._cpu_info_cached[0]} | "
            f"Physical CPU Cores: {self._cpu_info_cached[1]}"
        )
        # Get temperature
        if m.temp_samples:
            current_temp = m.temp_samples[-1]
            avg_temp = m.avg(m.temp_samples)
            table.add_row(
                f"Current Temp: {colorize(current_temp, THRESHOLDS.temp_warn, THRESHOLDS.temp_crit, '°C')} | "
                f"Average Temp: {colorize(avg_temp, THRESHOLDS.temp_warn, THRESHOLDS.temp_crit, '°C')}"
            )
        else:
            table.add_row("[yellow]No temperature data[/yellow]")
        # Get CPU frequency
        cpu_freq = psutil.cpu_freq()
        time.sleep(SETTINGS.sample_interval_sec)
        if cpu_freq:
            table.add_row(f"Current CPU Frequency: {int(cpu_freq.current)} MHz")

        return table


# === Signal Handling & Summary ===
def signal_handler(sig, frame):
    """Handle Ctrl-C for graceful shutdown."""
    monitor.stop()

def print_summary(m: CPUMonitor):
    """Print a final summary of CPU and temperature data."""
    console.print("\n⚜️")
    console.print("\n[bold green]Summary:[/bold green]")

    # Show start & finish times + duration
    if m.start_time and m.end_time:
        start_str = m.start_time.strftime("%Y-%m-%d %H:%M:%S")
        end_str = m.end_time.strftime("%Y-%m-%d %H:%M:%S")
        duration = m.end_time - m.start_time  # timedelta

        # Format duration as H:M:S
        total_seconds = int(duration.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        duration_str = f"{hours}h {minutes}m {seconds}s"

        console.print(f"[cyan]Monitoring started:[/cyan] {start_str}")
        console.print(f"[cyan]Monitoring finished:[/cyan] {end_str}")
        console.print(f"[cyan]Total duration:[/cyan] {duration_str}\n")

    if m.cpu_samples:
        console.print(f"Average CPU usage: {colorize(m.avg(m.cpu_samples), THRESHOLDS.cpu_warn, THRESHOLDS.cpu_crit, '%')}")
        console.print(f"Min CPU usage: {colorize(min(m.cpu_samples), THRESHOLDS.cpu_warn, THRESHOLDS.cpu_crit, '%')}")
        console.print(f"Max CPU usage: {colorize(max(m.cpu_samples), THRESHOLDS.cpu_warn, THRESHOLDS.cpu_crit, '%')}")
    else:
        console.print("[yellow]No CPU usage data collected.[/yellow]")

    if m.temp_samples:
        console.print(f"Average temperature: {colorize(m.avg(m.temp_samples), THRESHOLDS.temp_warn, THRESHOLDS.temp_crit, '°C')}")
        console.print(f"Min temperature: {colorize(min(m.temp_samples), THRESHOLDS.temp_warn, THRESHOLDS.temp_crit, '°C')}")
        console.print(f"Max temperature: {colorize(max(m.temp_samples), THRESHOLDS.temp_warn, THRESHOLDS.temp_crit, '°C')}")
    else:
        console.print("[yellow]No temperature data was collected.[/yellow]")


def print_random_colored_quote(quotes, colors):
    """Print a random quote in a random color."""
    quote = random.choice(quotes)
    color_code = random.choice(colors)
    print(f"\n{color_code}{quote}\033[0m")


def save_data_log(m: CPUMonitor):
    """Ask user if they want to save the CPU/temperature/frequency/load data to CSV."""
    choice = input("\nDo you want to save the collected data to CSV? [y/N]: ").strip().lower()
    if choice != "y":
        console.print("[yellow]Data was not saved.[/yellow]")
        return

    # Ask user for save location (default: current directory)
    default_dir = os.getcwd()
    outdir = input(f"Enter directory to save log [{default_dir}]: ").strip()
    if not outdir:
        outdir = default_dir
    os.makedirs(outdir, exist_ok=True)

    # Prepare filename with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"cpu_log_{timestamp}.csv"
    filepath = os.path.join(outdir, filename)

    try:
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Sample #", "Timestamp", "CPU Usage (%)", "Temperature (°C)", "Frequency (MHz)", "Load1", "Load5", "Load15"])
            for i in range(len(m.cpu_samples)):
                ts = m.timestamps[i] if i < len(m.timestamps) else ""
                cpu = f"{m.cpu_samples[i]:.1f}" if i < len(m.cpu_samples) else ""
                temp = f"{m.temp_samples[i]:.1f}" if i < len(m.temp_samples) else "N/A"
                freq = f"{m.freq_samples[i]:.1f}" if i < len(m.freq_samples) else ""
                load = m.load_samples[i] if i < len(m.load_samples) else ("", "", "")
                writer.writerow([i+1, ts, cpu, temp, freq, load[0], load[1], load[2]])
        console.print(f"[green]Data saved successfully to {filepath}[/green]")
    except Exception as e:
        console.print(f"[red]Failed to save data: {e}[/red]")


# === Main Execution ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CPU Monitoring Tool")
    parser.add_argument("--outdir", type=str, default=os.getcwd(),
                        help="Directory to save log file (default: current directory)")
    args = parser.parse_args()

    EXIT_QUOTES = [
    "There's no place like 127.0.0.1. - Anonymous",
    "It’s not a bug – it’s an undocumented feature. - Anonymous",
    "Data is the new oil. - Clive Humby",
    "Comments: the best way to tell yourself you forgot what you wrote. - Anonymous",
    "The most effective way to do it, is to do it. - Amelia Earhart",
    "Never trust a computer you can’t throw out a window. - Steve Wozniak",
    "Readability counts. - Tim Peters",
    "Errors should never pass silently - Unless explicitly silenced. - Tim Peters",
    "In the face of ambiguity, refuse the temptation to guess. - Tim Peters",
    "Thermal sensors: The CPU’s early warning system. - Anonymous",
    "Never let a computer know you're in a hurry. - John Carmack",
    "The problem with troubleshooting is that trouble shoots back. - Anonymous",
    "Fast, good, cheap: pick any two. - Anonymous",
    "A cool CPU is a happy CPU. - Anonymous",
    "Keep calm and cool your CPU. - Anonymous",
    "Talk is cheap. Show me the code. - Linus Torvalds",
    "Intelligence is the ability to avoid doing work, yet getting the work done. - Linus Torvalds",
    "Programming isn’t about what you know; it’s about what you can figure out. - Chris Pine",
    "Heat is a reminder that even machines have limits. - Anonymous",
    "Thermal throttling: When your CPU begs for mercy. - Anonymous",
    "Liquid cooling: Because sometimes air just isn’t enough. - Anonymous",
    "The Cloud is just someone else's computer. - Richard Stallman",
    "When I wrote this, only God and I understood what I was doing. Now, God only knows. - Anders Hejlsberg",
    "Code never lies, comments sometimes do. - Ron Jeffries",
    "Hardware: The parts of a computer you can kick. - Jeff Pesis",
    "There are two ways to write error-free programs; only the third one works. - Alan J. Perlis",
    "Any sufficiently advanced bug is indistinguishable from a feature. - Rich Kulawiec",
    "Overclocking: Because faster crashes are still crashes. - Anonymous",
    "If at first you don’t succeed, call it version 1.0. - Anonymous",
    "Fans: The unsung heroes keeping your CPU from becoming a toaster. - Anonymous",
    "Sometimes the best upgrade is just blowing the dust out. - Anonymous",
    "Thermal paste: The unsung hero between your CPU and sanity. - Anonymous",
    "The CPU: Where all your brilliant ideas get processed… eventually. - Anonymous",
    "A good CPU never sleeps, but it sure wishes it could. - Anonymous",
    "The CPU doesn’t care how smart you are—it just follows instructions. - Anonymous",
    "Every bit of CPU cycle counts… unless you’re procrastinating. - Anonymous",
    "A CPU’s job is simple: Think fast, don’t overheat. - Anonymous",
    "The CPU doesn’t judge your code—it just executes it, flaws and all. - Anonymous",
    "A CPU’s loyalty is to the instructions, not the programmer’s intent. - Anonymous",
    "The CPU obeys code, not your wishes. - Anonymous",
    "Heat is the silent enemy of performance. - Anonymous",
    "Fans spin so your CPU can think. - Anonymous",
    "Your intent is lost in translation between human brain and CPU cycles. - Anonymous",
    "Instructions speak louder than intentions to a CPU. - Anonymous",
    "The CPU follows commands, not hopes or assumptions. - Anonymous",
    "A hot CPU is a slow CPU in disguise. - Anonymous",
    "DevOps: Propagating errors in automated ways. - Kai Lentit"
    "Intentions require empathy; CPUs require precision. - Anonymous",
    "A computer will do what you tell it to do, but that may be much different from what you had in mind. - Anonymous",
    "Most good programmers do programming not because they expect to get paid or get adulation by the public, but because it is fun to program. - Linus Torvalds"
    ]
    COLORS = ["\033[35m"]  # Magenta

    monitor = CPUMonitor()
    ui = CPUUI(monitor)

    signal.signal(signal.SIGINT, signal_handler)
    monitor.start()

    try:
        with Live(console=console, refresh_per_second=4, transient=True) as live:
            while monitor.running.is_set():
                live.update(ui.make_renderable())
                time.sleep(0.1)  # Small UI refresh delay
    finally:
        monitor.stop()
        monitor.join()
        print_summary(monitor)
        if EXIT_QUOTES:
            print_random_colored_quote(EXIT_QUOTES, COLORS)
        save_data_log(monitor)
