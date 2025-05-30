import paramiko
import os
from pathlib import Path
from typing import Optional, Tuple
from large_scale_dimensionality_reduction.utils import setup_logger, cfg
import io

logger = setup_logger("ssh-logger")

class SSHClient:
    def __init__(self):
        """Initialize SSH client with credentials from environment variables."""
        if not all([cfg.HPC_HOST, cfg.HPC_USERNAME, cfg.HPC_SSH_KEY]):
            raise ValueError("HPC_HOST, HPC_USERNAME, and HPC_SSH_KEY environment variables must be set")
        
        self.host = cfg.HPC_HOST
        self.username = cfg.HPC_USERNAME
        self.key_path = os.path.expanduser(cfg.HPC_SSH_KEY)
        self.work_dir = cfg.HPC_WORK_DIR.replace("$USER", self.username)
        
        if not os.path.exists(self.key_path):
            raise ValueError(f"SSH key not found at {self.key_path}")
        
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
    def connect(self) -> None:
        """Establish SSH connection to the HPC server."""
        try:
            logger.info(f"Connecting to {self.username}@{self.host}")
            self.client.connect(
                hostname=self.host,
                username=self.username,
                key_filename=self.key_path
            )
            logger.info("Successfully connected to HPC server")
        except Exception as e:
            logger.error(f"Failed to connect to HPC server: {str(e)}")
            raise
            
    def disconnect(self) -> None:
        """Close the SSH connection."""
        if self.client:
            self.client.close()
            logger.info("Disconnected from HPC server")
            
    def upload_file(self, local_path: str, remote_path: str = None) -> str:
        """
        Upload a file to the HPC server.
        
        Args:
            local_path: Path to the local file
            remote_path: Optional remote path. If not provided, will use the same filename in work_dir
            
        Returns:
            str: The remote path where the file was uploaded
        """
        try:
            logger.info(f"Uploading {local_path} to {remote_path}")
            sftp = self.client.open_sftp()
            sftp.put(local_path, remote_path)
            sftp.close()
            logger.info("File upload successful")
            return remote_path
        except Exception as e:
            logger.error(f"Failed to upload file: {str(e)}")
            raise
            
    def execute_command(self, command: str) -> Tuple[int, str, str]:
        """
        Execute a command on the HPC server.
        
        Args:
            command: The command to execute
            
        Returns:
            Tuple[int, str, str]: (exit_code, stdout, stderr)
        """
        try:
            logger.info(f"Executing command: {command}")
            stdin, stdout, stderr = self.client.exec_command(command)
            exit_code = stdout.channel.recv_exit_status()
            stdout_str = stdout.read().decode()
            stderr_str = stderr.read().decode()
            
            if exit_code != 0:
                logger.error(f"Command failed with exit code {exit_code}: {stderr_str}")
            else:
                logger.info("Command executed successfully")
                
            return exit_code, stdout_str, stderr_str
        except Exception as e:
            logger.error(f"Failed to execute command: {str(e)}")
            raise 