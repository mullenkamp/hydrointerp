
import re
from cookielib import CookieJar
import urllib2

#-way to bypass the ssl certificate
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


class downloadMODIS():

    def __init__(self, username, password):
        # Create a password manager to deal with the 401 reponse that is returned from
        # Earthdata Login

        password_manager = urllib2.HTTPPasswordMgrWithDefaultRealm()
        password_manager.add_password(None, "https://urs.earthdata.nasa.gov", username, password)


        # Create a cookie jar for storing cookies. This is used to store and return
        # the session cookie given to use by the data server (otherwise it will just
        # keep sending us back to Earthdata Login to authenticate).  Ideally, we
        # should use a file based cookie jar to preserve cookies between runs. This
        # will make it much more efficient.

        cookie_jar = CookieJar()

        # Install all the handlers.
        opener = urllib2.build_opener(
            urllib2.HTTPBasicAuthHandler(password_manager),
            #urllib2.HTTPHandler(debuglevel=1),    # Uncomment these two lines to see
            #urllib2.HTTPSHandler(debuglevel=1),   # details of the requests/responses
            urllib2.HTTPCookieProcessor(cookie_jar))
        urllib2.install_opener(opener)


    def download(self, url, satType, cdate, tile, outf):
        # Create and submit the request. There are a wide range of exceptions that
        # can be thrown here, including HTTPError and URLError. These should be
        # caught and handled.

        year = cdate.year
        daynr = '%03d' %(cdate.timetuple().tm_yday)
        url = url + cdate.strftime('%Y.%m.%d') + '/'

        try:
            request = urllib2.Request(url)
            response = urllib2.urlopen(request)

            #-string to search for in the response
            s = satType + '.A' + str(year) + str(daynr) + '.' + tile + '.006.*.hdf'
            #-all files present in the response
            allFiles = response.read().splitlines()
            for f in allFiles:
                ss = re.findall(s, f)
                if ss:
                    ss = ss[0].split('"><img')[0]
                    break

            #-reconstruct the url using the first part and the exact match of the hdf file found in the list of files
            try:
                url = url + ss
                request = urllib2.Request(url)
                response = urllib2.urlopen(request)
                hdf = response.read()
                with open(outf, 'wb') as f:
                    f.write(hdf)
                return True
            except:
                return False
        except:
            return False











