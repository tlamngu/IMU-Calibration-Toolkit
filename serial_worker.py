# serial_worker.py
from PyQt6.QtCore import QThread, pyqtSignal
import serial
import serial.tools.list_ports
import time
import re
import numpy as np

class SerialWorker(QThread):
    serial_ports_signal = pyqtSignal(list)
    connection_status_signal = pyqtSignal(str, bool) # port_name, is_connected
    new_data_signal = pyqtSignal(str) # Raw line from serial
    log_message_signal = pyqtSignal(str)
    parsed_data_signal = pyqtSignal(dict) 
    def __init__(self, settings_manager):
        super().__init__()
        self.settings = settings_manager
        self.serial_port = None
        self.running = False
        self.port_to_connect = None
        self.baudrate_to_connect = self.settings.get("default_baudrate")

        self.accel_gyro_regex = re.compile(
            r"Time:(\d+).*?"
            r"GyroX:\s*(-?[\d.eE+-]+)\s*,?\s*"
            r"GyroY:\s*(-?[\d.eE+-]+)\s*,?\s*"
            r"GyroZ:\s*(-?[\d.eE+-]+)\s*,?\s*"
            r"AccX:\s*(-?[\d.eE+-]+)\s*,?\s*"
            r"AccY:\s*(-?[\d.eE+-]+)\s*,?\s*"
            r"AccZ:\s*(-?[\d.eE+-]+)\s*;"
        )
        self.mag_regex = re.compile(
            r"Time:(\d+).*?"
            r"MagX:\s*(-?[\d.eE+-]+)\s*,?\s*"
            r"MagY:\s*(-?[\d.eE+-]+)\s*,?\s*"
            r"MagZ:\s*(-?[\d.eE+-]+)\s*;"
        )

    def list_serial_ports(self):
        ports = sorted([port.device for port in serial.tools.list_ports.comports()])
        self.serial_ports_signal.emit(ports)

    def connect_to_port(self, port_name, baudrate=None):
        self.port_to_connect = port_name
        if baudrate:
            self.baudrate_to_connect = baudrate
        self.running = True
        if not self.isRunning():
            self.start() # Starts the run() method in a new thread
        else: # if already running, signal to try connecting
             pass # run loop will pick up self.port_to_connect

    def disconnect_port(self):
        self.running = False # Signal to stop
        # The run loop will handle closing the port

    def run(self):
        buffer = b''
        while True: # Outer loop for thread lifetime, allows reconnecting
            if self.port_to_connect and self.running:
                try:
                    self.log_message_signal.emit(f"Attempting to connect to {self.port_to_connect} at {self.baudrate_to_connect}...")
                    self.serial_port = serial.Serial(
                        port=self.port_to_connect,
                        baudrate=self.baudrate_to_connect,
                        timeout=0.1 # Non-blocking read
                    )
                    self.serial_port.reset_input_buffer()
                    self.serial_port.reset_output_buffer()
                    self.log_message_signal.emit(f"Connected to {self.port_to_connect}.")
                    self.connection_status_signal.emit(self.port_to_connect, True)
                    current_port_name = self.port_to_connect
                    self.port_to_connect = None # Consume connect request

                    while self.running and self.serial_port and self.serial_port.is_open:
                        try:
                            bytes_to_read = self.serial_port.in_waiting
                            if bytes_to_read > 0:
                                data = self.serial_port.read(bytes_to_read)
                                buffer += data
                            
                            while b'\n' in buffer:
                                line_bytes, buffer = buffer.split(b'\n', 1)
                                line_str = line_bytes.decode('utf-8', errors='ignore').strip()
                                if line_str:
                                    self.new_data_signal.emit(line_str) # Emit raw line
                                    self.parse_and_emit_data(line_str) # Also emit parsed data

                        except serial.SerialException as read_err:
                            self.log_message_signal.emit(f"Serial error during read: {read_err}")
                            break # Break inner loop to attempt cleanup/reconnect logic
                        except Exception as e:
                            self.log_message_signal.emit(f"Unexpected error in serial loop: {e}")
                            time.sleep(0.01) # Brief pause

                        time.sleep(0.001) # Small sleep to yield CPU, adjust as needed

                    # Exited inner read loop (either self.running is false or serial error)
                    if self.serial_port and self.serial_port.is_open:
                        self.serial_port.close()
                        self.log_message_signal.emit(f"Disconnected from {current_port_name}.")
                        self.connection_status_signal.emit(current_port_name, False)
                    self.serial_port = None

                except serial.SerialException as e:
                    self.log_message_signal.emit(f"Failed to connect to {self.port_to_connect}: {e}")
                    self.connection_status_signal.emit(self.port_to_connect or "N/A", False)
                    self.serial_port = None
                    self.port_to_connect = None # Reset connect attempt
                    time.sleep(1) # Wait before retrying if connect fails
                
            elif not self.running and self.serial_port: # If disconnect was called
                 if self.serial_port.is_open: self.serial_port.close()
                 self.log_message_signal.emit("Serial port closed by request.")
                 self.connection_status_signal.emit(self.serial_port.name or "N/A", False)
                 self.serial_port = None
                 # Thread will exit if self.running is False and no port_to_connect
            
            if not self.running and not self.port_to_connect:
                break # Exit the run method and thread finishes

            time.sleep(0.1) # Main loop sleep when idle or waiting for connect command
        self.log_message_signal.emit("Serial worker thread stopped.")

    def parse_and_emit_data(self, line_str):
        ag_match = self.accel_gyro_regex.search(line_str)
        m_match = self.mag_regex.search(line_str)
        
        # Initialize data dictionary for this line
        parsed_line_data = {'timestamp_ms': None, 'accel': None, 'gyro': None, 'mag': None}
        # Try to get a primary timestamp
        if ag_match:
            try:
                parsed_line_data['timestamp_ms'] = int(ag_match.group(1))
            except: pass
        elif m_match and parsed_line_data['timestamp_ms'] is None:
            try:
                parsed_line_data['timestamp_ms'] = int(m_match.group(1))
            except: pass
        
        if ag_match:
            try:
                g = np.array([float(ag_match.group(2)), float(ag_match.group(3)), float(ag_match.group(4))], dtype=float)
                a = np.array([float(ag_match.group(5)), float(ag_match.group(6)), float(ag_match.group(7))], dtype=float)
                if np.all(np.isfinite(a)) and np.all(np.isfinite(g)):
                    parsed_line_data['accel'] = a
                    parsed_line_data['gyro'] = g
            except Exception as e:
                self.log_message_signal.emit(f"Error parsing Accel/Gyro data: {e} from '{line_str[:30]}...'")

        if m_match:
            try:
                m = np.array([float(m_match.group(2)), float(m_match.group(3)), float(m_match.group(4))], dtype=float)
                if np.all(np.isfinite(m)):
                    parsed_line_data['mag'] = m
            except Exception as e:
                self.log_message_signal.emit(f"Error parsing Mag data: {e} from '{line_str[:30]}...'")

        if parsed_line_data['timestamp_ms'] is not None and \
           (parsed_line_data['accel'] is not None or parsed_line_data['mag'] is not None): # Must have at least accel or mag
            self.parsed_data_signal.emit(parsed_line_data)


    def stop(self):
        self.running = False
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
        self.quit() # Ask event loop to quit
        self.wait() # Wait for thread to finish