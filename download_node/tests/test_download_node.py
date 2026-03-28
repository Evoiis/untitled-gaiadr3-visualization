import pytest
import zmq
import threading
import time
import logging

import star_data_pb2

from node import DownloadNode

class MockDownloadWrapper():

    def __init__(self, string):
        self.string = string

    def get_data(self):
        return self.string

class MockDataProcessor():
    
    def process_data(self, data):
        data_req = star_data_pb2.DataRequest()
        data_req.timestamp = int(time.time())
        data_req.node_name = data
        return data_req.SerializeToString()


TEST_SERVER_PORT = 5556
TEST_STRING = "TEST STRING"
logger = logging.getLogger(__name__)

def test_download_node_req():
    mdw = MockDownloadWrapper(TEST_STRING)
    mdp = MockDataProcessor()
    dnode = DownloadNode(
        mdw, mdp, TEST_SERVER_PORT
    )
    dnode_thread = threading.Thread(
        target=dnode.run_node
    )
    dnode_thread.start()
    
    zmq_context = zmq.Context()
    socket = zmq_context.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, 5000) # recv timeout after 5 second

    logger.debug("Init Complete, now connect")
    socket.connect(f"tcp://localhost:{TEST_SERVER_PORT}")      
    
    data_req = star_data_pb2.DataRequest()
    data_req.timestamp = int(time.time())
    data_req.node_name = "Test Node"
    socket.send(data_req.SerializeToString())

    logger.debug("Sent waiting for reply.")
    
    try:
        message = socket.recv()
    except zmq.Again:
        dnode.stop_loop()
        dnode_thread.join()
        pytest.fail("Client recv timed out while waiting for a reply from server.")


    received_data_req = star_data_pb2.DataRequest()
    received_data_req.ParseFromString(message)

    logger.debug(f"Received Message: {received_data_req.node_name}")

    dnode.stop_loop()
    dnode_thread.join()
    assert received_data_req.node_name == TEST_STRING
