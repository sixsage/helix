import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

min_r0=1.0
max_r0=10.0
nlayers=10
sigma=0.01

# from: https://www-jlc.kek.jp/subg/offl/lib/docs/helix_manip/node3.html
# a track is parameterized by a 5dim helix
# MODIFIED TO BE A SIN 
# here it is expressed relative to an initial angle, phi0, and calculated and some updated angle phi
# hence it sweeps out the track as you vary phi and keep the other parameters fixed
def track(phi,d0,phi0,A,omega,dz,tanl):

    x = (d0+phi)*np.cos(phi0) 
    y = (d0+phi)*np.sin(phi0) + A*(np.sin(omega*(phi - phi0)))
    z = dz - tanl*phi
    return x,y,z

# For a given phi and track parameters, calculates the distance from a target r02
# used by find_phi to determine the angle that intersects with a fixed-radius circle detector
def dr(phi, r02,d0,phi0,A,omega,dz,tanl):

    # get the xyz of the track at this phi
    x,y,z = track(phi, d0,phi0,A,omega,dz,tanl)
    r2=x*x+y*y

    # get the distance from the target r02
    dr = np.fabs(r2-r02)

    return dr

# Find the phi value where the track intersects a given layer r
def find_phi(r0, d0,phi0,A,omega,dz,tanl):

    # this is lazy, but rather than inverting the equations we just minimize the distance
    res = scipy.optimize.minimize(dr,0,method='Nelder-Mead',args = (r0, d0,phi0,A,omega,dz,tanl))#, bounds =(0,1.0))

    return res.x[0]

# calculate the chisq between the track defined by the parameters and the spacepoints given
def chisq(params,x,y,z):
    ihit=0
    c2=0

    # find the hits for the track parameters
    for r0 in np.linspace(min_r0,max_r0,nlayers):
        phi0 = find_phi(r0*r0,*params)
        x0,y0,z0 = track(phi0,*params)

        # calculate deviation from observed hit
        c2 = c2 + (x0-x[ihit])**2 + (y0-y[ihit])**2 + (z0-z[ihit])**2   # assume equal uncertainty in x,y,z
        ihit = ihit+1

    return c2

# calculate the track parameters for a set of spacepoints
def fit_params(x,y,z):

    res = scipy.optimize.minimize(chisq,(0,0.69,0.5,0.5,0,0.1),args=(x,y,z),method='Nelder-Mead', bounds = ( (0,0.02),(0,2*np.pi),(0.1,1),(0.1,1.0),(-2.5,2.5),(-1.0,1.0)) )
    return res.x

def fast_fit_params(x, y, z):
    """
    x, y, z: np.array([])
    return:
    """
    # ref. 10.1016/0168-9002(88)90722-X
    r = x**2 + y**2
    u = x / r
    v = y / r
    # assuming the imapact parameter is small
    # the v = 1/(2b) - u x a/b - u^2 x epsilon x (R/b)^3
    pp, vv = np.polyfit(u, v, 2, cov=True)
    b = 0.5 / pp[2]
    a = -pp[1] * b
    R = np.sqrt(a**2 + b**2)
    e = -pp[0] / (R / b) ** 3  # approximately equals to d0
    dev = 2 * e * R / b**2

    magnetic_field = 2.0
    pT = magnetic_field * R  # in MeV
    # print(a, b, R, e, pT)

    p_rz = np.polyfit(np.sqrt(r), z, 2)
    pp_rz = np.poly1d(p_rz)
    z0 = pp_rz(abs(e))

    r3 = np.sqrt(r + z**2)
    p_zr = np.polyfit(r3, z, 2)
    cos_val = p_zr[0] * z0 + p_zr[1]
    theta = np.arccos(cos_val)
    eta = -np.log(np.tan(theta / 2.0))
    # phi = np.atan2(b, a)
    phi = np.arctan2(b, a)

    return e,phi,pT,z0,-1.0*eta

def fast_find_phi(r02, d0,phi0,A,omega,dz,tanl):

    ra2 = 0
    phia=0
    phib = 0.1
    xb,yb,zb= track(phib,d0,phi0,A,omega,dz,tanl)
    rb2 = xb*xb+yb*yb

    while (rb2-ra2>0.01):
        if (rb2>r02 and ra2<r02):
            phib = phia + (phib-phia)*(r02-ra2)/(rb2-ra2)
            xb,yb,zb= track(phib,d0,phi0,A,omega,dz,tanl)
            rb2 = xb*xb+yb*yb
        if (rb2<r02 and ra2<r02):
            phibnew = phia + (phib-phia)*(r02-ra2)/(rb2-ra2)
            phia = phib
            ra2 = rb2
            phib = phibnew
            xb,yb,zb= track(phib,d0,phi0,A,omega,dz,tanl)
            rb2 = xb*xb+yb*yb
        if (rb2>r02 and ra2>r02):
            phianew = phib + (phia-phib)*(r02-rb2)/(ra2-rb2)
            phib = phia
            rb2 = ra2
            phia = phianew
            xa,ya,za= track(phia,d0,phi0,A,omega,dz,tanl)
            ra2 = xa*xa+ya*ya

# find the intersections with the detector layers for these track parameters, add noise
# def make_hits(params):
#     xs=[]
#     ys=[]
#     zs =[]
    
#     for r0 in np.linspace(min_r0,max_r0,nlayers):
#         phi0 = find_phi(r0*r0,*params)
#         x0,y0,z0 = track(phi0,*params)
#         xs.append(x0+np.random.normal(scale=sigma))
#         ys.append(y0+np.random.normal(scale=sigma))
#         zs.append(z0+np.random.normal(scale=sigma))


#     return xs,ys,zs

def make_hits(params):
    xs=[]
    ys=[]
    zs =[]

    gaussianNoise=False
    asymmetricNoise = False
    if asymmetricNoise:
        skewness = 4
        for r0 in np.linspace(min_r0,max_r0,nlayers):
            phi0 = find_phi(r0*r0,*params)
            # print(" r0 = ",r0, " phi0 = ",phi0)
            # fphi0= fast_find_phi(r0*r0,*params)
            # print(" fr0 = ",r0, " fphi0 = ",fphi0)
            x0,y0,z0 = track(phi0,*params)

            xs.append(x0 + scipy.stats.skewnorm.rvs(a=skewness, scale=sigma))
            ys.append(y0 + scipy.stats.skewnorm.rvs(a=skewness, scale=sigma))
            zs.append(z0 + scipy.stats.skewnorm.rvs(a=skewness, scale=sigma))
    else:
        for r0 in np.linspace(min_r0,max_r0,nlayers):
            phi0 = find_phi(r0*r0,*params)
            # print(" r0 = ",r0, " phi0 = ",phi0)
            # fphi0= fast_find_phi(r0*r0,*params)
            # print(" fr0 = ",r0, " fphi0 = ",phi0)
            x0,y0,z0 = track(phi0,*params)

            # gaussian noise
            if (gaussianNoise):
                xs.append(x0+np.random.normal(scale=sigma))
                ys.append(y0+np.random.normal(scale=sigma))
                zs.append(z0+np.random.normal(scale=sigma))
            # use two gaussians, one wider
            else:
                if (np.random.random()>0.5):
                    xs.append(x0+np.random.normal(scale=sigma))
                    ys.append(y0+np.random.normal(scale=sigma))
                    zs.append(z0+np.random.normal(scale=sigma))
                else:
                    xs.append(x0+np.random.normal(scale=5*sigma))
                    ys.append(y0+np.random.normal(scale=5*sigma))
                    zs.append(z0+np.random.normal(scale=5*sigma))

    return xs,ys,zs

# generate random track parameters and the associated hits
def gen_tracks(n=1):
    
    tracks=[]
    for i in range(n):
        if (i%100==0):
            print("Track %d/%d" % (i,n))
        d0=np.fabs(np.random.normal(scale=0.01))
        phi=np.random.uniform(low=0,high=2*np.pi)
        A = np.random.uniform(low=0.5, high=1.5)
        omega = np.random.uniform(low=0.1, high=2.0)
        dz=np.random.normal(scale=1.0)
        tanl = np.random.normal(scale=0.3)
        params=(d0,phi,A,omega,dz,tanl)
        xs,ys,zs = make_hits(params)
        tracks.append([params,xs,ys,zs])
    return tracks
        

# scan each track parameter one at a time, make the hits and plot them
# just to verify the tracks look right
def scan():

    for d0 in np.linspace(0,0.25,10):
        params = (d0,0.0,1.0,1.0,0.0,0.2)
        xs,ys,zs = make_hits(params)
        plt.plot(xs,ys,"x",label="d0=%1.2f"%d0)
    plt.legend()
    plt.savefig("sinscan_d0.pdf")
    plt.clf()
    for phi0 in np.linspace(0,2*np.pi,10):
        params = (0.0,phi0,1.0,1.0,0.0,0.2)
        xs,ys,zs = make_hits(params)
        plt.plot(xs,ys,"x",label="phi0=%1.2f"%phi0)
    plt.legend()
    plt.savefig("sinscan_phi0.pdf")
    plt.clf()


    for A in np.linspace(0.1,1.0,10):
        params = (0.0,0.0,A,1.0,0.0,0.2)
        xs,ys,zs = make_hits(params)
        plt.plot(xs,ys,"x",label="A=%1.1f"%A)
    plt.legend()
    plt.savefig("sinscan_A.pdf")
    plt.clf()

    for omega in np.linspace(0.1,2.0,4):
        params = (0.0,0.0,1.0,omega,0.0,0.2)
        xs,ys,zs = make_hits(params)
        plt.plot(xs,ys,"x",label="omega=%1.1f"%omega)
    plt.legend()
    plt.savefig("sinscan_omega.pdf")
    plt.clf()

    for dz in np.linspace(-2.5,2.5,10):
        params = (0.0,0,1.0,1.0,dz,0.2)
        xs,ys,zs = make_hits(params)
        plt.plot(zs,ys,"x",label="dz=%1.1f"%dz)
    plt.legend()
    plt.savefig("sinscan_dz.pdf")
    plt.clf()

    for tanl in np.linspace(-1,1,10):
        params = (0.0,0,1.0,1.0,0,tanl)
        xs,ys,zs = make_hits(params)
        plt.plot(zs,ys,"x",label="tanl=%1.1f"%tanl)
    plt.legend()
    plt.savefig("sinscan_tanl.pdf")
    plt.clf()

# generate a track and fit it to recover the parameters
def test():

    params = [0,0.7,0.5,0.5,0,0.1]
    
    
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

    #
    plt.plot(x,y,".",color='green')
    plt.plot(xs,ys,"x",color='red')
    plt.plot(xf,yf,".",color='blue')
    plt.savefig("sin_plotxy.pdf")
    plt.clf()
    plt.plot(x,z,".")
    plt.plot(xs,zs,"x",color='red')
    plt.plot(x,z,".",color='blue')
    plt.savefig("sin_plotxz.pdf")


#test()

#scan()

# generate tracks and output them
tracks = gen_tracks(n=100000)
f=open("sintracks_100k_updated_non_gaussian_wider.txt","w")
for track in tracks:
    params = track[0]
    xs = track[1]
    ys = track[2]
    zs = track[3]
    f.write("%1.4f, %1.2f, %1.2f, %1.2f, %1.2f, %1.2f\n" % params)
    for i in range(len(xs)):
        f.write("%1.2f, %1.2f, %1.2f\n" % (xs[i],ys[i],zs[i]))
    f.write("EOT\n\n")
f.close()





