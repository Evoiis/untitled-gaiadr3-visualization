from src.gaia_data_processor import GaiaDataProcessor
from src.gaia_query import GaiaQueryParameters, GaiaQueryWrapper
from src.node import DownloadNode

def main():
    gqw = GaiaQueryWrapper(GaiaQueryParameters(
        n_stars_per_batch=1
    ))  # TODO: query launch params
    gdp = GaiaDataProcessor("data/") # TODO data folder path launch param

    dnode = DownloadNode(gqw, gdp, 5656, 1) # TODO port launch param

    dnode.run_node()


if __name__ == "__main__":
    main()
