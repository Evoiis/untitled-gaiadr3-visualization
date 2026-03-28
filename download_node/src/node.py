import zmq
import logging

import star_data_pb2

logger = logging.getLogger(__name__)

class DownloadNode():

    def __init__(self, download_wrapper, data_processor, server_port: int):
        """
        Input:
        - download_wrapper/data_processor, inject classes to get/process data with
        - server_port, port to bind the server socket to
        - client_target_port, port to send messages to
        """
        self.zmq_context = zmq.Context()
        self.socket = self.zmq_context.socket(zmq.REP)
        self.socket.setsockopt(zmq.RCVTIMEO, 1000) # recv timeout after 1 second
        self.socket.bind(f"tcp://*:{server_port}")

        self.download_wrapper = download_wrapper
        self.data_processor = data_processor
        self.data = None

        self.stop = False

    def _shutdown(self):
        pass

    def run_node(self):
        try:
            self._loop()
        except KeyboardInterrupt:
            pass
            
        logger.info("Shutting down Download Node")
        self._shutdown()
    
    def stop_loop(self):
        self.stop = True

    def _loop(self):

        while not self.stop:
            # Wait for requests from other nodes
            try:
                received = self.socket.recv()
            except zmq.Again:
                # Reached timeout, go back to loop start
                continue

            data_req = star_data_pb2.DataRequest()
            data_req.ParseFromString(received)

            logger.info(f"Received request at {data_req.timestamp} from {data_req.node_name}")

            if self.data is None:
                self.data = self.data_processor.process_data(self.download_wrapper.get_data())
            
            self.socket.send(self.data)

