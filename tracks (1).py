import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

min_r0=1.0
max_r0=10.0
nlayers=10
sigma=0.01

# from: https://www-jlc.kek.jp/subg/offl/lib/docs/helix_manip/node3.html
# a track is parameterized by a 5dim helix
# here it is expressed relative to an initial angle, phi0, and calculated and some updated angle phi
# hence it sweeps out the track as you vary phi and keep the other parameters fixed
def track(phi, d0,phi0,pt,dz,tanl):
    alpha = 1/2 # 1/cB
    q=1
    kappa = q/pt
    rho = alpha/kappa
    x = d0*np.cos(phi0) + rho*(np.cos(phi0)-np.cos(phi0+phi))
    y = d0*np.sin(phi0) + rho*(np.sin(phi0)-np.sin(phi0+phi))
    z = dz - rho*tanl*phi
    return x,y,z

# For a given phi and track parameters, calculates the distance from a target r02
# used by find_phi to determine the angle that intersects with a fixed-radius circle detector
def dr(phi, r02,d0,phi0,pt,dz,tanl):

    # get the xyz of the track at this phi
    x,y,z = track(phi, d0,phi0,pt,dz,tanl)
    r2=x*x+y*y

    # get the distance from the target r02
    dr = np.fabs(r2-r02)

    return dr

# Find the phi value where the track intersects a given layer r
def find_phi(r0, d0,phi0,pt,dz,tanl):

    # this is lazy, but rather than inverting the equations we just minimize the distance
    res = scipy.optimize.minimize(dr,0,method='Nelder-Mead',args = (r0, d0,phi0,pt,dz,tanl))#, bounds =(0,1.0))

    return res.x[0]

# calculate the chisq between the track defined by the parameters and the spacepoints given
def chisq(params,x,y,z):
    ihit=0
    c2=0

    # find the hits for the track parameters
    for r0 in np.linspace(min_r0,max_r0,nlayers):
        phi0 = find_phi(r0*r0,*params)
        x0,y0,z0 = track(phi0,*params)
        print(x0, y0, z0)
        # calculate deviation from observed hit
        c2 = c2 + (x0-x[ihit])**2 + (y0-y[ihit])**2 + (z0-z[ihit])**2   # assume equal uncertainty in x,y,z
        ihit = ihit+1

    return c2


# calculate the track parameters for a set of spacepoints (0,0.69,100.,0,0.1)
def fit_params(x,y,z):

    res = scipy.optimize.minimize(chisq,(0,0.69,100.,0,0.1),args=(x,y,z),method='Nelder-Mead', bounds = ( (0,0.02),(0,2*np.pi),(25,200),(-2.5,2.5),(-1.0,1.0)) )
    return res.x


# find the intersections with the detector layers for these track parameters, add noise
def make_hits(params):
    xs=[]
    ys=[]
    zs =[]
    
    for r0 in np.linspace(min_r0,max_r0,nlayers):
        phi0 = find_phi(r0*r0,*params)
        x0,y0,z0 = track(phi0,*params)
        xs.append(x0+np.random.normal(scale=sigma))
        ys.append(y0+np.random.normal(scale=sigma))
        zs.append(z0+np.random.normal(scale=sigma))


    return xs,ys,zs

# generate random track parameters and the associated hits
def gen_tracks(n=1):
    
    tracks=[]
    for i in range(n):
        if (i%1000==0):
            print("Track %d/%d" % (i,n))
        d0=np.fabs(np.random.normal(scale=0.01))
        phi=np.random.uniform(low=0,high=2*np.pi)
        pt=np.random.uniform(low=25,high=200)
        dz=np.random.normal(scale=1.0)
        tanl = np.random.normal(scale=0.3)
        params=(d0,phi,pt,dz,tanl)
        xs,ys,zs = make_hits(params)
        tracks.append([params,xs,ys,zs])
    return tracks
        

# scan each track parameter one at a time, make the hits and plot them
# just to verify the tracks look right
def scan():

    for d0 in np.linspace(0,0.25,10):
        params = (d0,0.0,25.0,0.0,0.2)
        xs,ys,zs = make_hits(params)
        plt.plot(xs,ys,"x",label="d0=%1.2f"%d0)
    plt.legend()
    plt.savefig("scan_d0.pdf")
    plt.clf()
    for phi0 in np.linspace(0,2*np.pi,10):
        params = (0.0,phi0,25.0,0.0,0.2)
        xs,ys,zs = make_hits(params)
        plt.plot(xs,ys,"x",label="phi0=%1.2f"%phi0)
    plt.legend()
    plt.savefig("scan_phi0.pdf")
    plt.clf()


    for pt in np.linspace(25,200,10):
        params = (0.0,0,pt,0.0,0.2)
        xs,ys,zs = make_hits(params)
        plt.plot(xs,ys,"x",label="pt=%1.1f"%pt)
    plt.legend()
    plt.savefig("scan_pt.pdf")
    plt.clf()

    for dz in np.linspace(-2.5,2.5,10):
        params = (0.0,0,25,dz,0.2)
        xs,ys,zs = make_hits(params)
        plt.plot(zs,ys,"x",label="dz=%1.1f"%dz)
    plt.legend()
    plt.savefig("scan_dz.pdf")
    plt.clf()

    for tanl in np.linspace(-1,1,10):
        params = (0.0,0,25,0,tanl)
        xs,ys,zs = make_hits(params)
        plt.plot(zs,ys,"x",label="tanl=%1.1f"%tanl)
    plt.legend()
    plt.savefig("scan_tanl.pdf")
    plt.clf()

# generate a track and fit it to recover the parameters
def test():

    params = [0,0.7,100.,0,0.1]
    
    
    hits=[]
    
    xs,ys,zs = make_hits(params)
    
    # get the full trace
    lastphi=np.arctan2(xs[-1],ys[-1])
    phi = np.linspace(0,lastphi,100)
    x,y,z = track(phi,*params)
    # try to fit the track
    fitparams = fit_params(xs,ys,zs)
    
    # get the full fitted trace
    xf,yf,zf = track(phi,*fitparams)
    
    print("True params = ",params)
    print("Fit  params = ",fitparams)

    
    # plt.plot(x,y,".",color='green')
    # plt.plot(xs,ys,"x",color='red')
    # plt.plot(xf,yf,".",color='blue')
    # plt.savefig("plotxy.pdf")
    # plt.clf()
    # plt.plot(x,z,".")
    # plt.plot(xs,zs,"x",color='red')
    # plt.plot(x,z,".",color='blue')
    # plt.savefig("plotxz.pdf")


# test()

# scan()

# generate tracks and output them
tracks = gen_tracks(n=100000)
f=open("tracks_100k.txt","w")
for track in tracks:
    params = track[0]
    xs = track[1]
    ys = track[2]
    zs = track[3]
    f.write("%1.2f, %1.2f, %1.2f, %1.2f, %1.2f\n" % params)
    for i in range(len(xs)):
        f.write("%1.2f, %1.2f, %1.2f\n" % (xs[i],ys[i],zs[i]))
    f.write("EOT\n\n")
f.close()



