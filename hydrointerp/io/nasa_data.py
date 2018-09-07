# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 11:32:02 2018

@author: MichaelEK
"""

from pydap.client import open_url
from pydap.cas.urs import setup_session

dataset_url = 'https://disc2.gesdisc.eosdis.nasa.gov:443/opendap/TRMM_L3/TRMM_3B42_Daily.7/2016/01/3B42_Daily.20160101.7.nc4'

dataset_url = 'https://gpm1.gesdisc.eosdis.nasa.gov:443/opendap/GPM_L3/GPM_3IMERGDF.05/2018/05/3B-DAY.MS.MRG.3IMERG.20180501-S000000-E235959.V05.nc4'

base_url_trimm = 'https://disc2.gesdisc.eosdis.nasa.gov:443/opendap/TRMM_L3'
base_url_gpm = 'https://gpm1.gesdisc.eosdis.nasa.gov:443/opendap/GPM_L3'

product_dict = {'trmm': {'daily': '3B42_Daily.7'}, 'gpm': {'daily': '3IMERGDF.05'}}

#session = setup_session('Dryden', 'NasaData4me', check_url=base_url_trimm)
#
#dataset = open_url(dataset_url, session=session)

username = 'Dryden'
password = 'NasaData4me'


d1 = dataset['precipitation'][:40]

#def session(username, password):
#        """
#        Function to create a session to log into the opendap server.
#
#        Parameters
#        ----------
#        username : str
#            Your username
#        password : str
#            Your password
#
#        Returns
#        -------
#        session object
#        """
#
#        session1 = setup_session(username, password)
#
#        return session1


class Gesdisc(object):
    """

    """
    def __init__(self, username, password, mission):
        if mission == 'trmm':
            self.base_url = base_url_trimm
        elif mission == 'gpm':
            self.base_url = base_url_gpm
        else:
            raise ValueError('mission should be either trmm or gpm')

        self.mission = mission

        self.session = setup_session(username, password, check_url=self.base_url)

    def close(self):
        self.session.close()


    def get_dataset_types(self):
        """
        Function to get all of the dataset types and associates attributes.

        Returns
        -------
        dict
        """
        url2 = '{mission}_{product}/2016/01/'.format(mission=self.mission.upper(), product=product_dict[self.mission]['daily'])
        if self.mission == 'gpm':
            file_name = '3B-DAY.MS.MRG.3IMERG.20160101-S000000-E235959.V05.nc4'
        elif self.mission == 'trmm':
            file_name = '3B42_Daily.20160101.7.nc4'

        full_url = self.base_url + '/' + url2 + file_name

        dataset = open_url(full_url, session=self.session)

        dataset_dict = {}
        for i in dataset:
            dataset_dict.update({i.name: i.attributes})

        return dataset_dict


    def get_data(self, dataset_type, time_freq, from_date, to_date, min_lat=None, max_lat=None, min_lon=None, max_lon=None):
        """

        """















g1 = Gesdisc(username, password, 'trmm')
d1 = g1.get_dataset_types()















