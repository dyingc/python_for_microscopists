# https://youtu.be/6gUdlygtscI
"""
Demonstration of GMM on 1D, 2D, and 3D data. 

For 1D
First we generate data by sampling random data from two normal distributions
Then, we decmpose it into 3 (or different number) gaussians. 
Finally, we plot the original data and the decomposed Gaussians. 

Do something similar for 2D and 3D cases... 
Generate data, perform GMM and plot individual components.

"""


from sklearn import mixture
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

# create some data
# Draw samples from different normal distributions so we get data that is good
#to demonstrate GMM. Different mean and Std. dev. 
#Concatenate to create a single data set
x = np.concatenate((np.random.normal(2, 5, 1000), np.random.normal(7, 3, 1000), np.random.normal(11, 2.5, 1000)))
plt.plot(x)

f=x.reshape(-1,1)

#We created data from two normal distributions but for the fun of it let us
# decompose our data into 3 Gaussians. n_components=3

# A function using "bic" to find the best number of components
def find_best_params_by_bic(data, n_components_range=None, grace_period=0):
    if n_components_range is None:
        n_components_range = range(1, len(data))
    global_lowest_bic = np.infty # The lowest bic across all covariance types
    best_n_component = n_components_range[0]
    best_type = "spherical"
    bics = []
    cv_types = ['tied', 'diag', 'full', 'spherical']
    for cv_type in cv_types:
        local_lowest_bic = np.infty # The lowest bic for this covariance type
        increasing_bic_count = 0  # count the number of times BIC increases consecutively
        for n_components in n_components_range:
            gmm = mixture.GaussianMixture(n_components=n_components, covariance_type=cv_type, random_state=42)
            gmm.fit(data)
            bic = gmm.bic(data)
            bics.append(bic)
            if bic < local_lowest_bic:
                local_lowest_bic = bic
                if local_lowest_bic < global_lowest_bic:
                    global_lowest_bic = local_lowest_bic
                    best_n_component = n_components
                    best_type = cv_type
                    best_gmm = gmm
                increasing_bic_count = 0  # reset the count if BIC decreases
                print(f"Found better model with BIC={local_lowest_bic} and n_components={n_components} and type={cv_type}")
            else:
                print(f"Wrose BIC found: BIC={bic} and n_components={n_components} and type={cv_type}")
                increasing_bic_count += 1  # increment the count if BIC increases
            # stop if BIC increases more than the grace period
            if increasing_bic_count > grace_period:
                print("\n\n")
                break # test the next covariance type
    print(f"Best model has BIC={global_lowest_bic} and n_components={best_n_component} and type={best_type}")
    return best_gmm, best_n_component, best_type, bics

best_gmm, best_n_component, best_type, bics = find_best_params_by_bic(f, n_components_range=range(1, len(f)), grace_period=0)

# Similar function of `find_best_params_by_bic` but using "aic"

def find_best_params_by_aic(data, n_components_range:None, grace_period=0):
    if n_components_range is None:
        n_components_range = range(1, len(data))
    global_lowest_aic = np.infty  # The lowest aic across all covariance types
    best_n_component = n_components_range[0]
    best_type = "spherical"
    aics = []
    cv_types = ['tied', 'diag', 'full', 'spherical']
    for cv_type in cv_types:
        local_lowest_aic = np.infty  # The lowest aic for this covariance type
        increasing_aic_count = 0  # count the number of times AIC increases consecutively
        for n_components in n_components_range:
            gmm = mixture.GaussianMixture(n_components=n_components, covariance_type=cv_type, random_state=42)
            gmm.fit(data)
            aic = gmm.aic(data)
            aics.append(aic)
            if aic < local_lowest_aic:
                local_lowest_aic = aic
                if local_lowest_aic < global_lowest_aic:
                    global_lowest_aic = local_lowest_aic
                    best_n_component = n_components
                    best_type = cv_type
                    best_gmm = gmm
                increasing_aic_count = 0  # reset the count if AIC decreases
                print(f"Found better model with AIC={local_lowest_aic} and n_components={n_components} and type={cv_type}")
            else:
                print(f"Wrose AIC found: AIC={aic} and n_components={n_components} and type={cv_type}")
                increasing_aic_count += 1  # increment the count if AIC increases
            # stop if AIC increases more than the grace period
            if increasing_aic_count > grace_period:
                print("\n\n")
                break  # test the next covariance type
    print(f"Best model has AIC={global_lowest_aic} and n_components={best_n_component} and type={best_type}")
    return best_gmm, best_n_component, best_type, aics

best_gmm, best_n_component, best_type, aics = find_best_params_by_aic(f, n_components_range=range(1, len(f)), grace_period=0)

g = mixture.GaussianMixture(n_components=best_n_component,covariance_type=best_type)
g.fit(f)
weights = g.weights_
means = g.means_
covars = g.covariances_



x_axis = x
x_axis.sort()

colors = ['red', 'green', 'blue']
def get_covars_element(covars, i):
    if i < len(covars):
        return covars[i]
    else:
        return covars[-1]
plt.hist(f, bins=100, histtype='bar', density=True, ec=colors[0], alpha=0.5)
for i, weight in enumerate(weights[:3]):
    plt.plot(x_axis,weight*stats.norm.pdf(x_axis,means[0],np.sqrt(get_covars_element(covars, i))).ravel(), c=colors[i])

plt.grid()
plt.show()

###########################

#2D example
# Parts of the code from
# https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html
########################

# Generate some data
from sklearn.datasets import make_blobs
import numpy as np
from matplotlib import pyplot as plt
X, y_true = make_blobs(n_samples=400, centers=4,
                       cluster_std=0.60, random_state=0)
X = X[:, ::-1] # flip axes for better plotting

rng = np.random.RandomState(13)
stretching_matrix = rng.randn(2, 2)
X_stretched = np.dot(X, stretching_matrix)
print(f"Before stretching: X.shape = {X.shape}, mean: {np.mean(X, axis=0)}, std: {np.std(X, axis=0)}")
print(f"After stretching: X_stretched.shape = {X_stretched.shape} with {stretching_matrix}, mean: {np.mean(X_stretched, axis=0)}, std: {np.std(X_stretched, axis=0)}")
plt.scatter(X_stretched[:, 0], X_stretched[:, 1], s=7, cmap='viridis')


from sklearn.mixture import GaussianMixture as GMM
a_best_gmm, a_best_n_component, a_best_type, aics = find_best_params_by_aic(X_stretched, n_components_range=range(1, len(X_stretched)), grace_period=0)
b_best_gmm, b_best_n_component, b_best_type, bics = find_best_params_by_bic(X_stretched, n_components_range=range(1, len(X_stretched)), grace_period=0)
gmm = GMM(n_components=b_best_n_component, covariance_type=b_best_type, random_state=42)


from matplotlib.patches import Ellipse
def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    
    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(xy=position, width=nsig * width, height=nsig * height, 
                             angle=angle, **kwargs))
        
        
def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=7, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=7, zorder=2)
    ax.axis('equal')
    
    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)       


plot_gmm(gmm, X_stretched)

##########################
# 3D
# https://github.com/sitzikbs/gmm_tutorial
##############

import numpy as np
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot as plt

#Generate 3D data with 4 clusters
 # set gaussian ceters and covariances in 3D
n_clusters = 4
n_features = 3
means = np.array([[0.5, 0.0, 0.0],
                      [0.0, 0.0, 0.0],
                      [-0.5, -0.5, -0.5],
                      [-0.8, 0.3, 0.4]])
means = np.random.normal(0, 1, (n_clusters, n_features))
covs = np.array([np.diag([0.01, 0.01, 0.03]),
                     np.diag([0.08, 0.01, 0.01]),
                     np.diag([0.01, 0.05, 0.01]),
                     np.diag([0.03, 0.07, 0.01])])
covs = np.random.normal(0.08, 0.02, (n_clusters, n_features, n_features))

n_gaussians = means.shape[0]  #Number of clusters

N = 1000 #Number of points to be generated for each cluster.
points = []
for i in range(len(means)):
    x = np.random.multivariate_normal(means[i], covs[i], N )
    points.append(x)
points = np.concatenate(points)

#Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:,0], points[:,1], points[:,2], s=1, alpha=1)
ax.view_init(35.246, 45)
plt.show()


#fit the gaussian model
a_best_gmm, a_best_n_component, a_best_type, aics = find_best_params_by_aic(points, n_components_range=range(1, len(X_stretched)), grace_period=0)
b_best_gmm, b_best_n_component, b_best_type, bics = find_best_params_by_bic(points, n_components_range=range(1, len(X_stretched)), grace_period=0)
gmm = GaussianMixture(n_components=max(1, int((b_best_n_component+b_best_n_component+0.5)/2)), covariance_type='diag')
gmm.fit(points)
diag_covars = np.zeros_like(gmm.covariances_)
for i in range(b_best_gmm.covariances_.shape[0]):
    diag_covars[i] = np.diag(np.diag(np.diag(b_best_gmm.covariances_[i]))) # 1. convert the covars matrix to diag matrix first and then convert it to a vector, by removing all the zero elements
b_best_gmm.covariances_ = diag_covars
# gmm = b_best_gmm # TODO


#Functions to visualize data
import matplotlib.cm as cmx

def plot_sphere(w=0, c=[0,0,0], r=[1, 1, 1], subdiv=20, ax=None, sigma_multiplier=3):
    '''
        plot a sphere surface
        Input: 
            c: list of n_feature (=3) elements, sphere center
            r: list of n_feature (=3) elements, sphere original scale in each axis ( allowing to draw elipsoids)
            subdiv: scalar, number of subdivisions (subdivision^2 points sampled on the surface)
            ax: optional pyplot axis object to plot the sphere in.
            sigma_multiplier: sphere additional scale (choosing an std value when plotting gaussians)
        Output:
            ax: pyplot axis object
    '''

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0.0:pi:complex(0,subdiv), 0.0:2.0 * pi:complex(0,subdiv)]
    x = sigma_multiplier*r[0] * sin(phi) * cos(theta) + c[0]
    y = sigma_multiplier*r[1] * sin(phi) * sin(theta) + c[1]
    z = sigma_multiplier*r[2] * cos(phi) + c[2]
    cmap = cmx.ScalarMappable()
    cmap.set_cmap('jet')
    c = cmap.to_rgba(w)

    ax.plot_surface(x, y, z, color=c, alpha=0.2, linewidth=1)

    return ax

def visualize_3d_gmm(points, w, mu, stdev, name:str):
    '''
    plots points and their corresponding gmm model in 3D
    Input: 
        points: N X 3, sampled points
        w: n_gaussians, gmm weights
        mu: 3 X n_gaussians, gmm means
        stdev: 3 X n_gaussians, gmm standard deviation (assuming diagonal covariance matrix)
    Output:
        None
    '''

    n_gaussians = mu.shape[1]
    N = int(np.round(points.shape[0] / n_gaussians))
    # Visualize data
    fig = plt.figure(figsize=(8, 8))
    axes = fig.add_subplot(111, projection='3d')
    axes_scale = [-1 * 3, 1 * 3]
    axes.set_xlim(axes_scale)
    axes.set_ylim(axes_scale)
    axes.set_zlim(axes_scale)
    plt.set_cmap('Set1')
    colors = cmx.Set1(np.linspace(0, 1, n_gaussians))
    for i in range(n_gaussians):
        idx = range(i * N, (i + 1) * N)
        axes.scatter(points[idx, 0], points[idx, 1], points[idx, 2], alpha=0.3, c=colors[i])
        plot_sphere(w=w[i], c=mu[:, i], r=stdev[:, i], ax=axes)

    plt.title(name)
    axes.set_xlabel('X')
    axes.set_ylabel('Y')
    axes.set_zlabel('Z')
    # axes.view_init(35.246, 45)
    axes.view_init(35.246*100, 45*100)
    plt.show()


visualize_3d_gmm(points, gmm.weights_, gmm.means_.T, np.sqrt(gmm.covariances_).T, name="gmm")
visualize_3d_gmm(points, b_best_gmm.weights_, b_best_gmm.means_.T, np.sqrt(b_best_gmm.covariances_).T, name="b_best_gmm")



