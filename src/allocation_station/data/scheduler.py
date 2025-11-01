"""Automated data updates and scheduling system."""

from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, time, timedelta
from pathlib import Path
import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR
from pydantic import BaseModel, Field
import logging
from enum import Enum


class ScheduleFrequency(str, Enum):
    """Schedule frequency options."""
    MINUTELY = "minutely"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    MARKET_OPEN = "market_open"
    MARKET_CLOSE = "market_close"


class UpdateJob(BaseModel):
    """Represents a scheduled update job."""
    job_id: str
    name: str
    frequency: ScheduleFrequency
    symbols: List[str]
    data_source: str = "yahoo"
    enabled: bool = True
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    success_count: int = 0
    error_count: int = 0
    last_error: Optional[str] = None


class DataScheduler:
    """
    Automated data update scheduler.

    Manages scheduled jobs for automatic market data updates,
    portfolio refreshes, and data maintenance tasks.
    """

    def __init__(
        self,
        data_provider,
        cache_dir: str = "data/cache",
        log_dir: str = "data/logs"
    ):
        """
        Initialize data scheduler.

        Args:
            data_provider: Market data provider instance
            cache_dir: Directory for cached data
            log_dir: Directory for log files
        """
        self.data_provider = data_provider
        self.cache_dir = Path(cache_dir)
        self.log_dir = Path(log_dir)

        # Create directories
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.logger = self._setup_logging()

        # Initialize scheduler
        self.scheduler = BackgroundScheduler()
        self.scheduler.add_listener(
            self._job_listener,
            EVENT_JOB_EXECUTED | EVENT_JOB_ERROR
        )

        # Job tracking
        self.jobs: Dict[str, UpdateJob] = {}

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger('DataScheduler')
        logger.setLevel(logging.INFO)

        # File handler
        log_file = self.log_dir / f"scheduler_{datetime.now().strftime('%Y%m%d')}.log"
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

        return logger

    def start(self):
        """Start the scheduler."""
        if not self.scheduler.running:
            self.scheduler.start()
            self.logger.info("Scheduler started")

    def stop(self):
        """Stop the scheduler."""
        if self.scheduler.running:
            self.scheduler.shutdown()
            self.logger.info("Scheduler stopped")

    def add_update_job(
        self,
        job_id: str,
        name: str,
        symbols: List[str],
        frequency: ScheduleFrequency,
        time_of_day: Optional[time] = None,
        data_source: str = "yahoo",
        callback: Optional[Callable] = None
    ) -> UpdateJob:
        """
        Add a scheduled data update job.

        Args:
            job_id: Unique job identifier
            name: Human-readable job name
            symbols: List of symbols to update
            frequency: Update frequency
            time_of_day: Specific time for daily/weekly updates
            data_source: Data source to use
            callback: Optional callback function after update

        Returns:
            UpdateJob instance
        """
        # Create job record
        job = UpdateJob(
            job_id=job_id,
            name=name,
            frequency=frequency,
            symbols=symbols,
            data_source=data_source
        )

        # Determine trigger
        trigger = self._create_trigger(frequency, time_of_day)

        # Create update function
        def update_function():
            self._execute_update(job_id, symbols, data_source, callback)

        # Schedule job
        self.scheduler.add_job(
            update_function,
            trigger=trigger,
            id=job_id,
            name=name,
            replace_existing=True
        )

        # Store job
        self.jobs[job_id] = job

        # Get next run time
        scheduled_job = self.scheduler.get_job(job_id)
        if scheduled_job:
            # APScheduler v3 uses 'next_run' instead of 'next_run_time'
            job.next_run = getattr(scheduled_job, 'next_run',
                                 getattr(scheduled_job, 'next_run_time', None))

        self.logger.info(f"Added update job: {name} (ID: {job_id})")

        return job

    def _create_trigger(
        self,
        frequency: ScheduleFrequency,
        time_of_day: Optional[time] = None
    ):
        """Create appropriate trigger for frequency."""
        if frequency == ScheduleFrequency.MINUTELY:
            return IntervalTrigger(minutes=1)

        elif frequency == ScheduleFrequency.HOURLY:
            return IntervalTrigger(hours=1)

        elif frequency == ScheduleFrequency.DAILY:
            if time_of_day:
                return CronTrigger(
                    hour=time_of_day.hour,
                    minute=time_of_day.minute
                )
            else:
                return IntervalTrigger(days=1)

        elif frequency == ScheduleFrequency.WEEKLY:
            # Monday at specified time (or 9:00 AM)
            hour = time_of_day.hour if time_of_day else 9
            minute = time_of_day.minute if time_of_day else 0
            return CronTrigger(day_of_week='mon', hour=hour, minute=minute)

        elif frequency == ScheduleFrequency.MONTHLY:
            # First day of month
            hour = time_of_day.hour if time_of_day else 9
            minute = time_of_day.minute if time_of_day else 0
            return CronTrigger(day=1, hour=hour, minute=minute)

        elif frequency == ScheduleFrequency.MARKET_OPEN:
            # US market open (9:30 AM ET) - weekdays only
            return CronTrigger(
                day_of_week='mon-fri',
                hour=9,
                minute=30,
                timezone='America/New_York'
            )

        elif frequency == ScheduleFrequency.MARKET_CLOSE:
            # US market close (4:00 PM ET) - weekdays only
            return CronTrigger(
                day_of_week='mon-fri',
                hour=16,
                minute=0,
                timezone='America/New_York'
            )

        else:
            raise ValueError(f"Unknown frequency: {frequency}")

    def _execute_update(
        self,
        job_id: str,
        symbols: List[str],
        data_source: str,
        callback: Optional[Callable] = None
    ):
        """Execute data update for job."""
        job = self.jobs.get(job_id)
        if not job or not job.enabled:
            return

        self.logger.info(f"Executing update job: {job.name} (symbols: {len(symbols)})")

        try:
            # Fetch data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)  # Last 30 days

            data = self.data_provider.fetch_historical_data(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date
            )

            # Update job status
            job.last_run = datetime.now()
            job.success_count += 1

            # Get next run time
            scheduled_job = self.scheduler.get_job(job_id)
            if scheduled_job:
                job.next_run = scheduled_job.next_run_time

            self.logger.info(
                f"Successfully updated {len(symbols)} symbols "
                f"(job: {job.name}, rows: {len(data)})"
            )

            # Execute callback if provided
            if callback:
                callback(data, job)

        except Exception as e:
            job.error_count += 1
            job.last_error = str(e)

            self.logger.error(
                f"Error updating job {job.name}: {e}",
                exc_info=True
            )

    def remove_job(self, job_id: str):
        """Remove a scheduled job."""
        if job_id in self.jobs:
            self.scheduler.remove_job(job_id)
            del self.jobs[job_id]
            self.logger.info(f"Removed job: {job_id}")

    def enable_job(self, job_id: str):
        """Enable a job."""
        if job_id in self.jobs:
            self.jobs[job_id].enabled = True
            self.scheduler.resume_job(job_id)
            self.logger.info(f"Enabled job: {job_id}")

    def disable_job(self, job_id: str):
        """Disable a job."""
        if job_id in self.jobs:
            self.jobs[job_id].enabled = False
            self.scheduler.pause_job(job_id)
            self.logger.info(f"Disabled job: {job_id}")

    def get_job_status(self, job_id: str) -> Optional[UpdateJob]:
        """Get status of a job."""
        return self.jobs.get(job_id)

    def get_all_jobs(self) -> List[UpdateJob]:
        """Get all scheduled jobs."""
        return list(self.jobs.values())

    def run_job_now(self, job_id: str):
        """Run a job immediately."""
        if job_id in self.jobs:
            job = self.scheduler.get_job(job_id)
            if job:
                job.modify(next_run_time=datetime.now())
                self.logger.info(f"Triggered immediate run for job: {job_id}")

    def _job_listener(self, event):
        """Listen to job events."""
        job_id = event.job_id

        if event.exception:
            self.logger.error(f"Job {job_id} failed: {event.exception}")
        else:
            self.logger.debug(f"Job {job_id} executed successfully")

    def add_portfolio_update_job(
        self,
        portfolio_name: str,
        symbols: List[str],
        frequency: ScheduleFrequency = ScheduleFrequency.DAILY,
        time_of_day: Optional[time] = None
    ) -> UpdateJob:
        """
        Add a portfolio update job.

        Args:
            portfolio_name: Name of portfolio
            symbols: Portfolio symbols
            frequency: Update frequency
            time_of_day: Time to run updates

        Returns:
            UpdateJob instance
        """
        job_id = f"portfolio_{portfolio_name}"
        name = f"Portfolio Update: {portfolio_name}"

        return self.add_update_job(
            job_id=job_id,
            name=name,
            symbols=symbols,
            frequency=frequency,
            time_of_day=time_of_day
        )

    def add_market_data_refresh_job(
        self,
        symbols: List[str],
        frequency: ScheduleFrequency = ScheduleFrequency.MARKET_CLOSE
    ) -> UpdateJob:
        """
        Add market data refresh job (runs at market close).

        Args:
            symbols: Symbols to refresh
            frequency: Update frequency

        Returns:
            UpdateJob instance
        """
        job_id = f"market_refresh_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        name = "Market Data Refresh"

        return self.add_update_job(
            job_id=job_id,
            name=name,
            symbols=symbols,
            frequency=frequency
        )

    def add_cache_cleanup_job(
        self,
        max_age_days: int = 30,
        frequency: ScheduleFrequency = ScheduleFrequency.WEEKLY
    ):
        """
        Add cache cleanup job.

        Args:
            max_age_days: Maximum age of cache files in days
            frequency: Cleanup frequency
        """
        job_id = "cache_cleanup"
        name = "Cache Cleanup"

        def cleanup_cache():
            self.logger.info("Running cache cleanup...")
            cutoff_date = datetime.now() - timedelta(days=max_age_days)
            removed_count = 0

            for file in self.cache_dir.glob('*'):
                if file.is_file():
                    mtime = datetime.fromtimestamp(file.stat().st_mtime)
                    if mtime < cutoff_date:
                        file.unlink()
                        removed_count += 1

            self.logger.info(f"Cache cleanup complete. Removed {removed_count} files.")

        trigger = self._create_trigger(frequency)

        self.scheduler.add_job(
            cleanup_cache,
            trigger=trigger,
            id=job_id,
            name=name,
            replace_existing=True
        )

        self.logger.info("Added cache cleanup job")

    def export_schedule(self, filename: str):
        """
        Export schedule configuration to file.

        Args:
            filename: Output filename
        """
        schedule_data = []

        for job in self.jobs.values():
            schedule_data.append(job.dict())

        df = pd.DataFrame(schedule_data)
        df.to_json(filename, orient='records', indent=2)

        self.logger.info(f"Exported schedule to {filename}")

    def import_schedule(self, filename: str):
        """
        Import schedule configuration from file.

        Args:
            filename: Input filename
        """
        df = pd.read_json(filename)

        for _, row in df.iterrows():
            self.add_update_job(
                job_id=row['job_id'],
                name=row['name'],
                symbols=row['symbols'],
                frequency=ScheduleFrequency(row['frequency']),
                data_source=row['data_source']
            )

        self.logger.info(f"Imported schedule from {filename}")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get scheduler statistics.

        Returns:
            Dictionary with scheduler statistics
        """
        total_jobs = len(self.jobs)
        enabled_jobs = sum(1 for job in self.jobs.values() if job.enabled)
        disabled_jobs = total_jobs - enabled_jobs

        total_success = sum(job.success_count for job in self.jobs.values())
        total_errors = sum(job.error_count for job in self.jobs.values())

        return {
            'total_jobs': total_jobs,
            'enabled_jobs': enabled_jobs,
            'disabled_jobs': disabled_jobs,
            'total_successful_runs': total_success,
            'total_errors': total_errors,
            'scheduler_running': self.scheduler.running,
            'jobs': [
                {
                    'job_id': job.job_id,
                    'name': job.name,
                    'enabled': job.enabled,
                    'last_run': job.last_run,
                    'next_run': job.next_run,
                    'success_count': job.success_count,
                    'error_count': job.error_count
                }
                for job in self.jobs.values()
            ]
        }
