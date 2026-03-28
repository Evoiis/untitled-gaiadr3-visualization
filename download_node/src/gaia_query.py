import os
import pandas as pd

from astroquery.gaia import Gaia
from astropy.table import Table

class GaiaQueryParameters:

    def __init__(
            self,
            parallax_lower_bound:int = 0.3,
            parallax_over_error_lower_bound: int = 5,
            ruwe_upper_bound: float = 1.4,
            phot_g_mean_mag_upper_bound: int = 15,
            n_stars: int = 500000,
            random_set: bool = True,
            guarantee_rad_velocity: bool = True
        ):
        self.parallax_lower_bound = parallax_lower_bound
        self.parallax_over_error_lower_bound = parallax_over_error_lower_bound
        self.ruwe_upper_bound = ruwe_upper_bound
        self.phot_g_mean_mag_upper_bound = phot_g_mean_mag_upper_bound
        self.n_stars = n_stars
        self.random_set = random_set
        self.guarantee_rad_velocity = guarantee_rad_velocity

class GaiaQueryWrapper:
   
    def __init__(self, query_parameters: GaiaQueryParameters, file_name: str = "", wr_to_file: bool= True):
        self._wr_to_file = wr_to_file
        self.qp = query_parameters
        self.file_name = file_name


    def get_data(self):
        file_read = self._read_from_file()
        if file_read is not None:
            return file_read
        
        else:

            data = self._send_gaia_query()
            df = data.to_pandas()

            self._write_to_file(df)

            return df

    def _read_from_file(self):
        if not self._wr_to_file:
            return None
 
        if not self.file_name:
            self.file_name = self._generate_file_name()
        
        if os.path.exists(self.file_name):
            return pd.read_csv(self.file_name)
        else:
            return None

    def _write_to_file(self, df: pd.DataFrame):
        if not self._wr_to_file:
            return

        if not self.file_name:
            self.file_name = self._generate_file_name()

        df.to_csv(self.file_name)

    def _generate_file_name(self):
        file_name = f"data_{self.qp.n_stars}"

        if self.qp.guarantee_rad_velocity:
            file_name += "_rv"

        if self.qp.random_set:
            file_name += "_ranset"

        return file_name + ".csv"
        

    def _send_gaia_query(
            self
        ) -> Table:
        """
        
        Output:
        Table with:
        source_id,ra,dec,parallax,pmra,pmdec,pmra_error,pmdec_error,phot_g_mean_mag,bp_rp,ruwe,radial_velocity,teff_gspphot,logg_gspphot,lum_flame,radius_flame
        """

        query = f"""
            SELECT TOP {self.qp.n_stars}
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

        if self.qp.random_set:
            query += " AND MOD(g.random_index, 50) = 0"

        job = Gaia.launch_job(query)
        return job.get_results()

