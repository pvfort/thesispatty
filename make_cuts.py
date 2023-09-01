#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import pandas as pd
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.coordinates import ICRS, Galactic, FK4, FK5  # Low-level frames
from astropy.coordinates import Angle, Latitude, Longitude  # Angles
import astropy.units as u
from astropy.visualization import simple_norm
from astropy.visualization import ZScaleInterval
from scipy.ndimage import gaussian_filter
from astropy.stats import mad_std
from astropy.stats import sigma_clipped_stats
from photutils.aperture import CircularAperture
from photutils.aperture import SkyCircularAperture
from photutils.aperture import aperture_photometry
import os


# In[ ]:


def choosef(RA, DEC, data):  # elige el campo donde estén las coordenadas
    # print(len(RA))

    COSMOS_NB = []
    for i in range(len(data)):
        field = data['Field'][i]
        R_min = data['RA_min'][i]
        R_max = data['RA_max'][i]
        D_min = data['DEC_min'][i]
        D_max = data['DEC_max'][i]

        for i in range(len(RA)):
            if RA[i] >= R_min and RA[i] <= R_max and DEC[i] >= D_min and DEC[i] <= D_max:
                COSMOS_NB.append([RA[i], DEC[i]])

    return COSMOS_NB  # retorna el field


# función cristobal
def random_apers(data):
    x = np.random.uniform(2200, 12000, 15000)
    y = np.random.uniform(2100, 10000, 15000)
    apers = CircularAperture([i for i in zip(x, y)], r=3.5/2)
    phot = aperture_photometry(data, apers)  # ,mask=kekazo)
    return phot

# función amanda


def get_names(coord):
    id_general = []
    for i in range(len(coord)):
        c = SkyCoord(ra=coord[i][0]*u.degree, dec=coord[i][1]*u.degree)
        ra = c.ra.hms
        dec = c.dec.dms
        h = round(ra[0]) < 10
        m1 = round(ra[1]) < 10
        s1 = round(ra[2]) < 10
        d = round(dec[0]) < 10
        m2 = round(dec[1]) < 10
        s2 = round(dec[2]) < 10
        lista = [[h, ra[0]], [m1, ra[1]], [s1, ra[2]],
                 [d, dec[0]], [m2, dec[1]], [s2, dec[2]]]
        contador = 0
        l = 'J'
        for j in lista:
            if contador <= 2:
                if j[0] == True:
                    l = l+'0'+str(int(round(j[1])))
                    contador += 1
                if j[0] == False:
                    l = l+str(int(round(j[1])))
                    contador += 1
            elif contador == 3:
                if j[0] == True:
                    l = l+'+0'+str(int(round(j[1])))
                    contador += 1
                if j[0] == False:
                    l = l+'+'+str(int(round(j[1])))
                    contador += 1
            elif contador >= 4:
                if j[0] == True:
                    l = l+'0'+str(int(round(j[1])))
                    contador += 1
                if j[0] == False:
                    l = l+str(int(round(j[1])))
                    contador += 1
        id_general.append(l)
    return id_general


def radec_minmax(hdr, coords): #guarda ra y dec mínimo y máximo
    wcs = WCS(hdr)
    c = [wcs.pixel_to_world(*coord) for coord in coords]
    ra_min = min([x.ra.deg for x in c])
    ra_max = max([x.ra.deg for x in c])
    
    dec_min = min([x.dec.deg for x in c])
    dec_max = max([x.dec.deg for x in c])
    
    return [ra_min, ra_max, dec_min, dec_max]


# In[ ]:


def cuts(coords, data, hdr, size, field, filter_used):
    # el path en el cual guardaremos las imágenes
    path = os.path.join('Cutouts', str(field)+'_'+str(filter_used))

    aper = random_apers(data)
    sums = aper['aperture_sum']
    clipped = sigma_clipped_stats(sums, sigma=2)
    sigmac = clipped[2]
    # nos retorna la desviación estandar que ocuparemos para que se vean bien los lya en los recortes
    print('Std: '+ str(sigmac))

    names = get_names(coords)  # función amanda

    for i in range(len(coords)):

        path_N = os.path.join(path, names[i]+'_'+str(filter_used))

        m_wcs = WCS(hdr)
        c = SkyCoord(coords[i][0], coords[i][1], frame=FK5, unit="deg")
        print(c)
        recorte = Cutout2D(data, c, (size*u.arcsec, size*u.arcsec), wcs=m_wcs)
        im = (recorte.data)
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        #norm = simple_norm(im, max_percent=99.9, min_percent=1)
        ax.imshow(im, origin='lower', interpolation='nearest',
                  cmap='Greys', vmin=-1*sigmac, vmax=3*sigmac)  # , norm = norm)
        #plt.scatter((len(recorte.data)/2), len(recorte.data) /
                    #2-9, marker='|', color='red', s=300)
        #plt.scatter((len(recorte.data)/2)-10, len(recorte.data) /
                    #2, marker='_', color='red', s=300)
        #plt.scatter((len(recorte.data)/2), (len(recorte.data)/2), s=300, facecolors='none', edgecolors='r')
        plt.axis('off')
        #plt.savefig(path_N, transparent=False, facecolor='white')

        plt.show()
        print(path_N)

    return


# In[ ]:


def cuts_mult(coords, data_filters, hdr_filters, names_filters, size, field):

    #data_filters = datag,datar,datai,dataz,datay, datanb
    #hdr_filters = hdrg, hdrr, hdri, hdrz, hdry, hdrnb
    # el path en el cual guardaremos las imágenes
    path = os.path.join('Cutouts', str(field))

    aper = random_apers(data_filters[-1])
    sums = aper['aperture_sum']
    clipped = sigma_clipped_stats(sums, sigma=2)
    sigmac = clipped[2]
    # nos retorna la desviación estandar que ocuparemos para que se vean bien los lya en los recortes
    print(sigmac)

    names = get_names(coords)  # función amanda

    for i in range(len(coords)):

        path_N = os.path.join(path, names[i])

        fig, ax = plt.subplots(2, 5, figsize=(20, 8))

        for x, data in enumerate(data_filters):
           # print(x,(x+1)//5, x-((x+1)//5 *5))
            m_wcs = WCS(hdr_filters[x])
            c = SkyCoord(coords[i][0], coords[i][1], frame=FK5, unit="deg")
            recorte = Cutout2D(
                data, c, (size*u.arcsec, size*u.arcsec), wcs=m_wcs)
            im = (recorte.data)

            ax[(x)//5, x-((x)//5 * 5)].imshow(im, origin='lower', interpolation='nearest',
                                              cmap='Greys', vmin=-1*sigmac, vmax=3*sigmac)  # , norm = norm)
            ax[(x)//5, x-((x)//5 * 5)].scatter((len(recorte.data)/2),
                                               len(recorte.data)/2-9, marker='|', color='red', s=300)
            ax[(x)//5, x-((x)//5 * 5)].scatter((len(recorte.data)/2) -
                                               10, len(recorte.data)/2, marker='_', color='red', s=300)
            ax[(x)//5, x-((x)//5 * 5)].set_title(names_filters[x], fontsize=18)
            ax[(x)//5, x-((x)//5 * 5)].axis('off')

        ax[1, 1].set_axis_off()
        ax[1, 2].set_axis_off()
        ax[1, 3].set_axis_off()
        ax[1, 4].set_axis_off()

        #plt.savefig(path_N, transparent=False, facecolor='white')
        fig.tight_layout()
        plt.show()
        # print(path_N)

    return

