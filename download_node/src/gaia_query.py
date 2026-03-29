import logging
import os
import time
import pandas as pd

from astroquery.gaia import Gaia
from astropy.table import Table

logger = logging.getLogger(__name__)

class GaiaQueryParameters:

    def __init__(
            self,
            parallax_lower_bound:float = 0.3,
            parallax_over_error_lower_bound: float = 5.0,
            ruwe_upper_bound: float = 1.4,
            phot_g_mean_mag_upper_bound: int = 15,
            n_stars_per_batch: int = 100000,
            use_random_set: bool = True,
            random_set_modulo: int = 50,
            guarantee_rad_velocity: bool = True,
        ):
        """
            parallax_lower_bound : Minimum parallax (mas); filters for nearer stars.
            parallax_over_error_lower_bound : Minimum parallax SNR (parallax / error).
            ruwe_upper_bound : Maximum RUWE; filters poor astrometric fits.
            phot_g_mean_mag_upper_bound : Maximum G magnitude (brighter stars only).
            n_stars_per_batch : Target number of stars per query.
            use_random_set : Enable deterministic subsampling via `random_index`.
            random_set_modulo : Controls sampling fraction (random_index % random_set_modulo).
                - 
            guarantee_rad_velocity : Require non-null radial velocity.
        """
        
        self.parallax_lower_bound = parallax_lower_bound
        self.parallax_over_error_lower_bound = parallax_over_error_lower_bound
        self.ruwe_upper_bound = ruwe_upper_bound
        self.phot_g_mean_mag_upper_bound = phot_g_mean_mag_upper_bound
        self.n_stars_per_batch = n_stars_per_batch
        self.use_random_set = use_random_set
        self.guarantee_rad_velocity = guarantee_rad_velocity
        self.random_set_modulo = random_set_modulo


class GaiaQueryWrapper:
   
    def __init__(
            self,
            query_parameters: GaiaQueryParameters,
            file_name: str = "",
            wr_to_file: bool= True
        ):
        self._wr_to_file = wr_to_file
        self.qp = query_parameters
        self.file_name = file_name

    def get_data(self, number_of_batches):
        return self.get_batches(number_of_batches)
    
    def get_batches(self, number_of_batches: int):
        """
        Makes multiple queries to Gaia Archives to get a larger dataset.
        Gaia Archives returns around 250000 stars maximum per query
        """

        if number_of_batches > 1 and not self.qp.use_random_set:
            raise Exception("use_random_set parameter must be true to download multiple batches")
        
        if number_of_batches <= 0 or number_of_batches >= self.qp.random_set_modulo:
            raise Exception(f"number_of_batches must be a postiive number less than random_set_modulo: {number_of_batches=}, {self.qp.random_set_modulo=}")
        
        file_read = self._read_from_file(number_of_batches)
        if file_read is not None:
            logging.info("Found existing data locally.")
            return file_read

        df = pd.DataFrame()

        for batch_num in range(number_of_batches):
            data = self._send_gaia_query(batch_num)
            
            df = pd.concat([df, data.to_pandas()], ignore_index=True).drop_duplicates(subset='source_id')

            # Wait to space out queries to Gaia Archive
            time.sleep(5)
        
        self._write_to_file(df, number_of_batches)

        return df            

    def _read_from_file(self, n_batches: int = 1):
        if not self._wr_to_file:
            return None
 
        if not self.file_name:
            self.file_name = self._generate_file_name(n_batches)
        
        if os.path.exists(self.file_name):
            return pd.read_csv(self.file_name)
        else:
            return None

    def _write_to_file(self, df: pd.DataFrame, n_batches: int = 1):
        if not self._wr_to_file:
            return

        if not self.file_name:
            self.file_name = self._generate_file_name(n_batches)

        df.to_csv(self.file_name)

    def _generate_file_name(self, n_batches: int):
        file_name = f"data_{self.qp.n_stars_per_batch}"

        if self.qp.guarantee_rad_velocity:
            file_name += "_rvelo"

        if self.qp.use_random_set:
            file_name += f"_ranset_modulo{self.qp.random_set_modulo}"

        if n_batches > 1:
            file_name += f"_batches{n_batches}"

        return file_name + ".csv"
        

    def _send_gaia_query(
            self,
            batch_num: int = 0
        ) -> Table:
        """
        Input:
            - batch_num : Batch index for `random_index` partitioning (random_index % random_set_modulo) == batch_num
        
        Output:
        Table with:
            source_id,ra,dec,parallax,pmra,pmdec,pmra_error,pmdec_error,phot_g_mean_mag,bp_rp,ruwe,radial_velocity,teff_gspphot,logg_gspphot,lum_flame,radius_flame
        """

        query = f"""
            SELECT TOP {self.qp.n_stars_per_batch}
                g.source_id,
                g.ra, g.dec,
                g.parallax,
                g.pmra, g.pmdec,
                g.pmra_error, g.pmdec_error,
                g.phot_g_mean_mag,
                g.bp_rp,
                g.ruwe,
                g.radial_velocity,
                g.teff_gspphot,
                g.logg_gspphot,
                ap.lum_flame,
                ap.radius_flame,
                h.original_ext_source_id
            FROM gaiadr3.gaia_source g
            LEFT JOIN gaiadr3.astrophysical_parameters ap
                ON g.source_id = ap.source_id
            LEFT JOIN gaiadr3.hipparcos2_best_neighbour h
                ON g.source_id = h.source_id
            WHERE
                g.parallax > {self.qp.parallax_lower_bound}
                AND g.parallax_over_error > {self.qp.parallax_over_error_lower_bound}
                AND g.ruwe < {self.qp.ruwe_upper_bound}
                AND g.phot_g_mean_mag < {self.qp.phot_g_mean_mag_upper_bound}
                AND g.bp_rp IS NOT NULL                
        """
        if self.qp.guarantee_rad_velocity:
            query += " AND g.radial_velocity IS NOT NULL"

        if self.qp.use_random_set:
            query += f" AND MOD(g.random_index, {self.qp.random_set_modulo}) = {batch_num}"

        job = Gaia.launch_job(query)
        return job.get_results()

