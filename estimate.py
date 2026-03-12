from core.layer import Layer
from core.model import Model
from ml_tools.optimizers import Optimizer, Nesterov_Accelerated_Gradient, Gradient_Descent, RMS_Propagation, ADAM
from ml_tools.initialisations import he_normal
from utils.data import create_normalized_data, process_df_2_output

from utils.logger import Logger

import plotly.graph_objects as go 
import sys

logger = Logger()

def main():

    if len(sys.argv) != 2:
        logger.error("python estimate.py *path_to_dataset*")
        return 1
    
    to_estimate_path: str = sys.argv[1]

    df = create_normalized_data(file_path=to_estimate_path)
    l_data = process_df_2_output(df)
    logger.info("dataset ready")

    try:
        model = Model()

        model.load_network("network_save")
        logger.info("model ready")
        for d in l_data:
            out = model.fire.full(d.get("data"), model.network.weights, model.network.biaises)
            logger.info(f"{out} {d.get("label")}")
    except Exception as exc:
        if str(exc): logger.error(exc)
        return 1


if __name__ == "__main__":
    exit(main())