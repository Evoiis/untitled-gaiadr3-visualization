import zmq
import logging

import star_data_pb2

class DownloadNode():

    def __init__(self, download_wrapper, data_processor, server_port: int, preload_data: bool = False, n_batches: int = 1):
        """
        Input:
        - download_wrapper/data_processor, inject classes to get/process data with
        - server_port, port to bind the server socket to
        - client_target_port, port to send messages to
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initializing Download Node, {server_port=}")

        self.zmq_context = zmq.Context()
        self.socket = self.zmq_context.socket(zmq.REP)
        self.socket.setsockopt(zmq.RCVTIMEO, 1000) # recv timeout after 1 second
        self.socket.bind(f"tcp://*:{server_port}")

        self.download_wrapper = download_wrapper
        self.data_processor = data_processor
        self.data = None

        self.stop = False
        self.n_batches = n_batches

        if preload_data:
            self.get_and_process_data()

    def _shutdown(self):
        pass

    def get_and_process_data(self):
        if self.data is None:
            self.data = self.data_processor.process_data(self.download_wrapper.get_data(self.n_batches))
        return self.data
    
    def get_motion_data(self):
        pass

    def run_node(self):
        try:
            self._loop()
        except KeyboardInterrupt:
            pass
            
        self.logger.info("Download Node: Shutting down")
        self._shutdown()
    
    def stop_loop(self):
        self.stop = True

    def _loop(self):
        self.logger.info("Download Node: Loop Starting")
        while not self.stop:
            try:
                # Wait for requests from other nodes
                received = self.socket.recv()
            except zmq.Again:
                # Reached timeout, go back to loop start
                continue

            data_req = star_data_pb2.DataRequest()
            data_req.ParseFromString(received)

            self.logger.info(f"Received request at {data_req.timestamp} from {data_req.node_name}. DT: {data_req.star_data_type}")

            if data_req.star_data_type == star_data_pb2.StarDataType.STAR_DATA:
                data = self.get_and_process_data()
            elif data_req.star_data_type == star_data_pb2.StarDataType.MOTION_STAR_DATA:
                # data = ?
            
            self.socket.send(data)
