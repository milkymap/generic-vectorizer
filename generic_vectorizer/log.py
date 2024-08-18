import logging 

logging.basicConfig(
    format='%(asctime)s - %(filename)s - %(levelname)s - %(lineno)3d - %(message)s -',
    level=logging.INFO
)

logger = logging.getLogger(name='grpc-llama')