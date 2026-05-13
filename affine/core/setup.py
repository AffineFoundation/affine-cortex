import os
import sys
import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv(override=True)

NETUID = 120
TRACE = 5

# Add custom TRACE level
logging.addLevelName(TRACE, "TRACE")


def _trace(self, msg, *args, **kwargs):
    if self.isEnabledFor(TRACE):
        self._log(TRACE, msg, args, **kwargs)


logging.Logger.trace = _trace
logger = logging.getLogger("affine")


def _get_component_name() -> str:
    """Identify component name from command line arguments.

    Used to pick the log directory: ``/var/log/affine/{component}/``.
    Anything not in the known set falls back to ``affine``.
    """
    if len(sys.argv) > 1:
        component = sys.argv[-1].lower()
        valid_components = ["api", "monitor", "scorer", "teacher", "validator"]
        if component in valid_components:
            return component
    return "affine"


class AbsoluteDayRotatingFileHandler(TimedRotatingFileHandler):
    """
    Custom handler that rotates logs every N days based on absolute dates.
    This ensures rotation happens at the same time regardless of service restarts.
    """
    def __init__(self, filename, interval_days=3, backupCount=20, encoding='utf-8', utc=True):
        # Use 'D' (days) as the base unit, but we'll override shouldRollover
        super().__init__(
            filename,
            when='D',
            interval=interval_days,
            backupCount=backupCount,
            encoding=encoding,
            utc=utc
        )
        self.interval_days = interval_days
        self.suffix = "%Y-%m-%d"
        # Prevent default .log extension in rotated files
        self.namer = lambda default_name: default_name.replace('.log.', '.')
        
    def shouldRollover(self, record):
        """
        Determine if rollover should occur based on absolute day count.
        Rotates every N days from epoch (1970-01-01), ensuring consistent rotation
        regardless of when the service was started.
        """
        # Get current time
        if self.utc:
            current_time = datetime.now(timezone.utc)
        else:
            current_time = datetime.now()
        
        # Calculate days since epoch
        if self.utc:
            epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
        else:
            epoch = datetime(1970, 1, 1)
        
        days_since_epoch = (current_time - epoch).days
        current_rotation_period = days_since_epoch // self.interval_days
        
        # Check if file exists and get its rotation period
        if os.path.exists(self.baseFilename):
            file_mtime = os.path.getmtime(self.baseFilename)
            if self.utc:
                file_time = datetime.fromtimestamp(file_mtime, tz=timezone.utc)
            else:
                file_time = datetime.fromtimestamp(file_mtime)
            
            file_days_since_epoch = (file_time - epoch).days
            file_rotation_period = file_days_since_epoch // self.interval_days
            
            # Rotate if we're in a new rotation period
            return current_rotation_period > file_rotation_period
        
        return False


def _setup_file_handler(component: str, level: int) -> logging.Handler:
    """
    Setup log file handler with rotation for the specified component.
    
    Log directory structure: /var/log/affine/{component}/
    Log file: {component}.log
    Rotation policy: every 3 days based on absolute dates
    - API component: keep 0 backups (only current 3-day period)
    - Other components: keep 20 backups (60 days)
    File suffix format: %Y-%m-%d
    
    The rotation is based on absolute date calculations (days since epoch),
    ensuring logs rotate consistently every 3 days regardless of service restarts.
    """
    log_dir = Path(f"/var/log/affine/{component}")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"{component}.log"
    
    backup_count = 1 if component == "api" else 10
    
    # Create custom handler with absolute 3-day rotation
    handler = AbsoluteDayRotatingFileHandler(
        log_file,
        interval_days=3,
        backupCount=backup_count,
        encoding="utf-8",
        utc=True  # Use UTC time for consistent rotation across timezones
    )
    
    handler.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    
    return handler


def _silence_noisy_loggers():
    """Silence noisy third-party library loggers"""
    noisy_loggers = [
        "websockets",
        "bittensor",
        "bittensor-cli",
        "btdecode",
        "asyncio",
        "aiobotocore.regions",
        "aiobotocore.credentials",
        "botocore",
        "httpx",
        "httpcore",
        "docker",
        "urllib3",
    ]
    
    for name in noisy_loggers:
        logging.getLogger(name).setLevel(logging.WARNING)
    
    # affinetes has its own handler, disable propagation to avoid duplicate logs
    affinetes_logger = logging.getLogger("affinetes")
    affinetes_logger.setLevel(logging.WARNING)
    affinetes_logger.propagate = False


def setup_logging(verbosity: int, component: str = None):
    """
    Setup logging system.
    
    Args:
        verbosity: Log level (0=SILENT, 1=INFO, 2=DEBUG, 3=TRACE)
        component: Component name (optional, defaults to auto-detection from sys.argv)
    """
    # Determine log level
    level_map = {
        0: logging.CRITICAL + 1,  # Silent
        1: logging.INFO,
        2: logging.DEBUG,
        3: TRACE,
    }
    level = level_map.get(verbosity, logging.INFO)
    
    # Get component name (use provided component or auto-detect)
    if component is None:
        component = _get_component_name()
    
    # Configure root logger (console output)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True
    )
    
    # Add file handler (log rotation)
    try:
        file_handler = _setup_file_handler(component, level)
        logging.getLogger().addHandler(file_handler)
        logger.info(f"Log file: /var/log/affine/{component}/{component}.log")
    except Exception as e:
        logger.warning(f"Failed to create log file: {e}")
    
    # Silence noisy third-party loggers
    _silence_noisy_loggers()
    
    # Set affine logger level
    logging.getLogger("affine").setLevel(level)
    
    # Set uvicorn logger (only for API service)
    if component == "api":
        logging.getLogger("uvicorn").setLevel(logging.INFO)
        logging.getLogger("uvicorn.access").setLevel(logging.INFO)


def info():
    """Shortcut: Set INFO level logging"""
    setup_logging(1)


def debug():
    """Shortcut: Set DEBUG level logging"""
    setup_logging(2)


def trace():
    """Shortcut: Set TRACE level logging"""
    setup_logging(3)