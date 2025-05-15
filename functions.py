import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import xarray as xr
import scipy
import cartopy.crs as ccrs
from IPython import get_ipython
import json
import pandas as pd


ice_dir = '/glade/work/jiangzhu/data/inputdata/cesm2_21ka/'
modern_psl_dir = '/glade/collections/cdg/data/CMIP6/CMIP/NCAR/CESM2/piControl/r1i1p1f1/Amon/psl/gn/files/d20190320/'
dir_100 = "/glade/campaign/cesm/community/palwg/LastGlacialMaximum/CESM2/b.e21.B1850CLM50SP.f09_g17.21ka.01"
dir_20 = "/glade/work/kkoepnick/cases/b.e21.B1850C5.f09_g17.21ka.20years"
atm_ext = '/atm/proc/tseries/month_1/'
land_ext = '/lnd/proc/tseries/month_1/'
atm_dir_100 = dir_100 + atm_ext
lnd_dir_100 = dir_100 + land_ext
lnd_dir_20 = dir_20 + land_ext
atm_dir_20 = dir_20 + atm_ext

modern_psl = xr.open_dataset(modern_psl_dir+ 'psl_Amon_CESM2_piControl_r1i1p1f1_gn_090001-099912.nc')['psl']
psl_all = xr.open_dataset(atm_dir_100 + 'b.e21.B1850CLM50SP.f09_g17.21ka.01.cam.h0.PSL.040101-050012.nc')['PSL']

psl = xr.open_dataset('data/psl_100.nc')

laurentide_smb = xr.open_dataset('data/laurentide_qice_no_mask_100.nc')
laurentide_smb_mask = xr.open_dataset('data/laurentide_qice_with_mask_100.nc')
laurentide_qice_melt =  xr.open_dataset('data/laurentide_qice_melt_no_mask_100.nc')
laurentide_qice_melt_mask =  xr.open_dataset('data/laurentide_qice_melt_with_mask_100.nc')

eurasian_smb = xr.open_dataset('data/eurasian_qice_no_mask_100.nc')
eurasian_smb_mask = xr.open_dataset('data/eurasian_qice_with_mask_100.nc')
eurasian_qice_melt =  xr.open_dataset('data/eurasian_qice_melt_no_mask_100.nc')
eurasian_qice_melt_mask =  xr.open_dataset('data/eurasian_qice_melt_with_mask_100.nc')

greenland_smb = xr.open_dataset('data/greenland_qice_no_mask_100.nc')
greenland_qice_melt =  xr.open_dataset('data/greenland_qice_melt_no_mask_100.nc')


def nans_3d_to_2d(original_matrix):
    '''
    Desc: 
        Turns a 3d array (time, lat, long) that contains nans into a 2d array that doesn't contain nans (lat*long, time)
        Allows us to do PCA without nans!
    Inputs (units): 
        original_matrix: the 3d matrix (time, lat, long) to convert
    Outputs (units):
        X: the 2d array (lat*long, time) that doesn't have any nans
        nvalid: the amount of non-nan values in the array
        valid_longs_flattened: the array storing all longitudes that don't have nans (in order)
        valid_lats_flattened: the array storing all latitudes that don't have nans (in order)
    '''
    time_length, lat_length, long_length = get_lengths(original_matrix)
    n = 0
    valid_longs_flattened = np.zeros(long_length * lat_length).astype("int")
    valid_lats_flattened = np.zeros(long_length *lat_length).astype("int")

    for long in range(0, long_length):
        for lat in range(0, lat_length):
            if ~np.isnan(original_matrix[0, lat, long]):
                valid_longs_flattened[n] = long
                valid_lats_flattened[n] = lat
                n = n + 1
    print("finished finding nans")
    print("total nans", lat_length*long_length - n)
    nvalid = n

    X = np.zeros((time_length, nvalid))

    for t in range(0, time_length):
        for n in range(0, nvalid):
            X[t, n] = original_matrix[t, valid_lats_flattened[n], valid_longs_flattened[n]]
        if t%50 == 0:
            print("up through", t)
            
    X = X.T
    print("X shape", X.shape)
    return X, valid_longs_flattened, valid_lats_flattened, long_length, lat_length


def nans_2d_to_3d(V, long_length, lat_length, valid_longs_flattened, valid_lats_flattened):
    '''
    Desc: 
        Turns a 2d array (location, time) that contains no nans into a 3d array that contains nans (lat, long, eigenvector value)
        Allows us to project the eigenvectors onto longitude and latitude by reinserting nans!
    Inputs (units): 
        V: the eigenvectors to reshape
        long_length: how many longitudes to project onto
        lat_length: how many latitudes to project onto
        valid_longs_flattened: the location of all non-nan longitudes within the original 3d array
        valid_lats_flattened: the location of all non-nan latitudes within the original 3d array
    Outputs (units):
        PCAs: a 3d array (lat, long, eigenvector) our eigenvectors, V, projected back onto latitude/longitude
    '''
    PCAs = np.zeros((lat_length, long_length, V.shape[1]))*np.nan
    for e in range(V.shape[1]):
        for n in range(V.shape[0]):
            PCAs[valid_lats_flattened[n], valid_longs_flattened[n],  e] = V[n, e]
    return PCAs


def standardize(matrix):
    
    std = np.std(matrix, axis=0)
    std[std == 0] = 1 
    
    return (matrix - np.mean(matrix, axis = 0)) / std


def pca(matrix, amount, stand=True):
    
    if (stand==True):
        matrix = standardize(matrix)
    
    cov = (matrix.values @ matrix.T.values) / (len(matrix[0]) - 1)
    
    print("covariance matrix shape", cov.shape)
    eigenvalues, V = scipy.sparse.linalg.eigs(cov, amount)
    
    V = np.real(V)
    eigenvalues = np.real(eigenvalues)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    V = V[:, idx]
    
    print("we calculated the first ", len(eigenvalues), " evals")
    
    return eigenvalues, V, matrix


def get_lengths(matrix):
    
    time_length = matrix.sizes['time']
    lat_length = matrix.sizes['lat'] 
    long_length = matrix.sizes['lon'] 
    
    print("time", time_length, "lat", lat_length,"long",  long_length)
    return time_length, lat_length, long_length

def plot_eigen(eigenvalues, title):
    
    plt.plot(eigenvalues)
    plt.xlabel("eigenvalue number (sorted)")
    plt.ylabel("eigenvalue")
    plt.title(f'Eigenvalues Values for {title} PCA')
    plt.show()
    
    percs = []
    for i in range(4):
        perc = np.round(eigenvalues[i]*100/sum(eigenvalues), 2)
        print(str(i+1)+" variance explained: "
              + str(perc) + "%")
        percs.append(perc)
        
    return percs


def plot_cov(sigma, title):
    
    sigmas_melt_squared = [s**2 for s in sigma]
    plt.plot(sigmas_melt_squared[:10])
    plt.xlabel("number (sorted)")
    plt.ylabel("singular value")
    plt.title(f'Singular Values for {title} MCA')
    plt.show()
    
    covariance_explained = []
    for i in range(4):
        ex = np.round(sigmas_melt_squared[i]*100/sum(sigmas_melt_squared), 2)
        print(str(i+1)+" covariance explained: "
              + str(ex)
              + "%")
        covariance_explained.append(ex)
        
    return covariance_explained


def mca_variance_explained(X, Y, U, V_t, num):

    X = X.values
    Y = Y.values
    
    a = U.T @ X 
    b = V_t @ Y 

    total_variance_X = np.sum(np.diag(X.T @ X)) / X.shape[1]  
    total_variance_Y = np.sum(np.diag(Y.T @ Y )) / Y.shape[1]  
    
    mode_k_variance_X = []
    mode_k_variance_Y = []
    
    for k in range(num):
        mode_k_variance_X.append((a[k, :] @ a[k, :].T) / X.shape[1])  
        mode_k_variance_Y.append((b[k, :] @ b[k, :].T) / Y.shape[1]) 

    var_k_percents_X = 100 * (mode_k_variance_X / total_variance_X)
    var_k_percents_Y = 100 * (mode_k_variance_Y / total_variance_Y)


    return var_k_percents_X, var_k_percents_Y


def plot_time_series(V,X,n, title, perc):
    
    time_series = V.T @ X.values
    time_x = np.arange(0,time_series.shape[1],1)/12
    
    fig, axs = plt.subplots((n+1)//2, 2, figsize=(20,10), dpi=200)
    axs = axs.flatten()
    
    for i in range(n):
        axs[i].set_xlabel('Time (yrs)', fontsize=18)
        axs[i].set_ylabel('Variance', fontsize=18)
        axs[i].plot(time_x, time_series[i,:])
        axs[i].set_title(f'Time Series for Mode {i+1} ({perc[i]}%)', fontsize=24)
        axs[i].fill_between(time_x, time_series[i,:], where=(time_series[i,:] >= 0), color='red', alpha=0.5)
        axs[i].fill_between(time_x, time_series[i,:], where=(time_series[i,:] < 0), color='blue', alpha=0.5)
        
    fig.suptitle(title, fontsize=36)
    plt.tight_layout()
    plt.show()
    return


def deseasonalize(data):
    
    monthly_avg = data.groupby("time.month").mean()
    deseasonalized_data = data.groupby("time.month") - monthly_avg
    return deseasonalized_data


def create_greendland_mask(data):
    
    greenland_mask = xr.full_like(data, 1)
    greenland_mask.loc[dict(lat=slice(59, 84), lon=slice(305, 350))] = np.nan
    greenland_mask.loc[dict(lat=slice(73, 82), lon=slice(295, 305))] = np.nan
    greenland_mask.loc[dict(lat=slice(73, 78), lon=slice(286, 295))] = np.nan
    return greenland_mask


def create_ice_mask(data):
    
    pct_ice = xr.open_dataset(ice_dir + 'surfdata_fv09_hist_16pfts_nourb_CMIP6_21ka.c200624.nc')["PCT_GLACIER"]
    ice_mask = np.tile(xr.where(pct_ice == 100, 1, np.nan), (data.shape[0], 1, 1))
    return ice_mask


def preprocess_laurentide(data, ice_mask=None): 
    
    print("converting data to m/s")
    data_m_per_year = data*60*60*24*365/1000
    
    print("deseasonalizing data")
    desesonalized_data = deseasonalize(data_m_per_year) 
    if ice_mask is not None:
        print("adding ice mask")
        desesonalized_data = desesonalized_data * ice_mask
        
    print("slicing data")
    desesonalized_data_sliced = data_m_per_year.sel(
        lat=slice(30, 90), lon=slice(190, 315))
    
    greenland_mask = create_greendland_mask(data)
    data_l = desesonalized_data_sliced * greenland_mask

    rocky_mask = xr.full_like(data_l, 1)
    rocky_mask.loc[dict(lat=slice(30, 46), lon=slice(240, 254))] = np.nan
    rocky_mask.loc[dict(lat=slice(42, 48), lon=slice(236, 241))] = np.nan
    data_laurentide = data_l * rocky_mask
    return data_laurentide

def preprocess_laurentide_no_des(data): 

    desesonalized_data = data 
        
    print("slicing data")
    desesonalized_data_sliced = desesonalized_data.sel(
        lat=slice(30, 90), lon=slice(190, 315))
    return desesonalized_data_sliced 

def preprocess_laurentide_only_dec(data): 
    
    print("converting data to m/s")
    data_m_per_year = data*60*60*24*365/1000
    
    print("deseasonalizing data")
    desesonalized_data = deseasonalize(data_m_per_year) 
    
    print("slicing data")
    desesonalized_data_sliced = data_m_per_year.sel(
        lat=slice(30, 90), lon=slice(190, 315))
    
    greenland_mask = create_greendland_mask(data)
    data_l = desesonalized_data_sliced * greenland_mask

    rocky_mask = xr.full_like(data_l, 1)
    rocky_mask.loc[dict(lat=slice(30, 46), lon=slice(240, 254))] = np.nan
    rocky_mask.loc[dict(lat=slice(42, 48), lon=slice(236, 241))] = np.nan
    data_laurentide = data_l * rocky_mask
    return data_laurentide

def preprocess_greenland(data, ice_mask=None): 
    
    print("converting data to m/s")
    data_m_per_year = data*60*60*24*365/1000
    
    print("deseasonalizing data")
    desesonalized_data = deseasonalize(data_m_per_year) 
    if ice_mask is not None:
        print("adding ice mask")
        desesonalized_data = desesonalized_data * ice_mask
        
    print("slicing data")

    desesonalized_data_sliced = desesonalized_data.sel(
        lat=slice(58, 90), lon=slice(286, 350))

    small_mask = xr.full_like(desesonalized_data_sliced, 1)
    small_mask.loc[dict(lat=slice(58, 75), lon=slice(285, 300))] = np.nan
    small_mask.loc[dict(lat=slice(81, 90), lon=slice(290, 295))] = np.nan
    small_mask.loc[dict(lat=slice(82, 92), lon=slice(295, 299))] = np.nan
    small_mask.loc[dict(lat=slice(80, 90), lon=slice(285, 290))] = np.nan
    small_mask.loc[dict(lat=slice(79, 80), lon=slice(280, 289))] = np.nan
    small_mask.loc[dict(lat=slice(58, 68), lon=slice(335, 350))] = np.nan

    masked = desesonalized_data_sliced * small_mask
    return masked

def preprocess_eurasia(data, ice_mask=None):
    print("converting data to m/s")
    data_m_per_year = data*60*60*24*365/1000
    
    print("deseasonalizing data")
    deseasonalized_data = deseasonalize(data_m_per_year) 
    if ice_mask is not None:
        print("adding ice mask")
        deseasonalized_data = deseasonalized_data * ice_mask
        
    print("slicing data")
    deseasonalized_data_new_coords = deseasonalized_data.assign_coords(
        lon=((deseasonalized_data.lon + 180) % 360) - 180).sortby("lon")
    desesonalized_data_sliced = deseasonalized_data_new_coords.sel(
        lat=slice(50, 90), lon=slice(-12, 115))

    small_mask = xr.full_like(desesonalized_data_sliced, 1)
    small_mask.loc[dict(lat=slice(60, 90), lon=slice(-12, -5))] = np.nan

    masked = desesonalized_data_sliced * small_mask
    return masked


def preprocess_na_ocean(data):
    
    print("converting to hecta pascals")
    hecta_data = data/100

    print("deseasonalizing")
    deseasonalized_data  = hecta_data #deseasonalize(data)
    
    print("slicing along lat=(20,80) and long=(-90,40)")
    deseasonalized_data_new_coords = deseasonalized_data.assign_coords(
        lon=((deseasonalized_data.lon + 180) % 360) - 180).sortby("lon")
    na_ocean_data = deseasonalized_data_new_coords.sel(lat=slice(20, 80), lon=slice(-90, 40))
    
    return na_ocean_data


def preprocess_na_ocean_small(data):
    
    print("converting to hecta pascals")
    hecta_data = data/100

    print("deseasonalizing")
    deseasonalized_data = deseasonalized_psl = deseasonalize(data)
    
    print("slicing along lat=(20,80) and long=(-90,40)")
    deseasonalized_data_new_coords = deseasonalized_data.assign_coords(
        lon=((deseasonalized_data.lon + 180) % 360) - 180).sortby("lon")
    na_ocean_data = deseasonalized_data_new_coords.sel(lat=slice(20, 70), lon=slice(-90, 40))
    
    return na_ocean_data



def svd(X,Y, amount):
    
    print("standardizing")
    stand_X = standardize(X)
    stand_Y = standardize(Y)
    
    print("creating covariance matrix")
    C = (stand_X.values @ stand_Y.T.values) / (len(stand_X[0]) - 1) # time dimension (the in common part!) 
    
    print("doing svd")
    U, Sigma, V_T = scipy.sparse.linalg.svds(C, k = amount)
    Sigma_sorted = Sigma[::-1]
    U_sorted = U[:, ::-1]
    V_T_sorted = V_T[::-1, :]
    
    
    return U_sorted, Sigma_sorted, V_T_sorted
    


def plot_pcs(data, n, title):
    reshaped_pcs, original, label, shift, l_override, perc = (
        data['reshaped_pca'],
        data['original'],
        data['label'],
        data['shift'],
        data['l_override'],
        data['perc']
    )
    
    fig, ax = plt.subplots((n + 1) // 2, 2, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(22, 15), dpi=200)
    axs = ax.flatten()
    
    for i in range(n):
        data = reshaped_pcs[:, :, i] 
        l = np.max(np.abs(np.nan_to_num(data, nan=0)))
        if i in l_override:
            l = l_override[i]

        plot_data_on_map(fig, axs, data, original, shift, l, i, label, perc[i], "PCA")        

    fig.suptitle(title, fontsize=36)
    plt.tight_layout()
    plt.show()


def plot_mca(data_1, data_2, n, title, cov, x_var, y_var):
    reshaped_mca_1, original_1, label_1, subtitle_1, shift_1, sigma_1, l_override_1 = (
        data_1['reshaped_mca'],
        data_1['original'],
        data_1['label'],
        data_1['subtitle'],
        data_1['shift'],
        data_1['sigma'],
        data_1['l_override']

    )
    reshaped_mca_2, original_2, label_2, subtitle_2, shift_2, sigma_2, l_override_2 = (
        data_2['reshaped_mca'],
        data_2['original'],
        data_2['label'],
        data_2['subtitle'],
        data_2['shift'],
        data_2['sigma'],
        data_2['l_override']

    )
    print("understood data")
    
    fig, ax = plt.subplots(n, 2, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(10,16), dpi=200)
    axs = ax.flatten()
    print("made plots")
    
    for i in range(n):
        to_plot_1 = reshaped_mca_1[:, :, i] * sigma_1[i]
        to_plot_2 = reshaped_mca_2[:, :, i] * sigma_2[i]

        print("plotting", i)
        
        l_1 = np.max(np.abs(np.nan_to_num(to_plot_1, nan=0)))
        l_2 = np.max(np.abs(np.nan_to_num(to_plot_2, nan=0)))

        if i in l_override_1:
            l_1 = l_override_1[i]
        if i in l_override_2:
            l_2 = l_override_2[i]

        label_x = str(cov[i]) + "% cov, "+ str(np.round(x_var[i],2))+ "% var"
        label_y = str(cov[i]) + "% cov, "+ str(np.round(y_var[i],2))+ "% var"

            
        plot_data_on_map(fig,axs,to_plot_1, original_1, shift_1, l_1, 2*i, label_1, label_x, "MCA", subtitle_1)
        plot_data_on_map(fig,axs,to_plot_2, original_2, shift_2, l_2, 2*i + 1, label_2, label_y, "MCA", subtitle_2)
        
    fig.suptitle(title, fontsize=36)
    plt.tight_layout()
    print("accessorizing")
    plt.show()


def plot_data_on_map(fig, axs, data, original, shift, l, i, label, perc, type, subtitle=None):        
        contour_plot = axs[i].contourf(
            original.lon,
            original.lat,
            data,
            cmap='bwr',
            levels=np.linspace(-l, l, 30)
        )
        axs[i].coastlines()
    
        if type == "MCA":
            axs[i].set_title(f"MCA mode {i // 2 + 1} for {subtitle} ({perc})", fontsize=12)
        else:
            axs[i].set_title(f"PCA mode {i + 1} ({perc}%)", fontsize=24)

        coords = original.assign_coords(lon=(original.lon - shift))
        axs[i].set_xticks(coords.lon[::20])
        axs[i].set_yticks(coords.lat[::20])
        axs[i].set_xticklabels([f'{lon:.2f}' for lon in coords.lon[::20]], fontsize=12)
        axs[i].set_yticklabels([f'{lat:.1f}' for lat in coords.lat[::20]], fontsize=12)
    
        cbar = fig.colorbar(contour_plot, ax=axs[i], orientation='horizontal', fraction=0.05, pad=0.1)
        cbar.set_label(label, fontsize=18)
        cbar.set_ticks([-l, 0, l])