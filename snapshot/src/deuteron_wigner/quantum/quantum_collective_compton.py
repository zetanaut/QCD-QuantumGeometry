"""One-free-quark-loop two-photon reference, not QCD/hadron matching.

All six cyclic charged-pion/photon orderings, zero-invariant wave subtraction,
linked Born/inverse/normalization and potential-linked scalar response.
Nc=3, equal quark masses, electric e^2 and external mode Yukawas stripped
from the box. Metric +---; no gluon or collective loops or resummation.
"""
from itertools import permutations, combinations, product
from functools import lru_cache
import numpy as np

from .canonical_lf_fields import GAMMA, slash
from .chiral_soft_matching import G5
from .quantum_pair_threshold import quadrature, feynman_nonlocal
from .quantum_collective_current import triangle_unit, bubble_vertex_norm
from .transverse_source_matching import dot, METRIC, ALPHA_EM, HBARC


def _slash_many(v):
    return np.einsum('...m,mij->...ij',v*METRIC,GAMMA)


def _trace_chain(vertices,propagators):
    result=vertices[0]@propagators[0]
    for i in range(1,4): result=result@vertices[i]@propagators[i]
    return np.trace(result,axis1=-2,axis2=-1)


def box_numerator(shifts,mass,vertices,*,method='wick'):
    """Angular-averaged original trace = A+B*l^2+C*l^4, d=4.

    Momentum-independent evanescent local terms cancel in the SAME wave
    subtraction. Wick contractions and an independent exact spherical
    fourth-degree cubature are separate evaluation routes, not two fits.
    shifts has shape (...,4 propagators,4 Lorentz components).
    """
    shifts=np.asarray(shifts,float)
    n=[_slash_many(shifts[...,i,:])+mass*np.eye(4) for i in range(4)]
    a=_trace_chain(vertices,n)
    if method=='wick':
        b=np.zeros_like(a)
        for i,j in combinations(range(4),2):
            for mu,sign in enumerate(METRIC):
                replaced=n.copy();replaced[i]=replaced[j]=GAMMA[mu]
                b+=sign*_trace_chain(vertices,replaced)/4
        c=0j
        for mu,nu in product(range(4),repeat=2):
            for labels in [(mu,mu,nu,nu),(mu,nu,mu,nu),(mu,nu,nu,mu)]:
                c+=METRIC[mu]*METRIC[nu]*_trace_chain(vertices,[GAMMA[i] for i in labels])/24
        return a,b,c
    if method!='angular': raise ValueError('Wick or independent angular trace required.')
    # 8 axis nodes + 16 hypercube corners, equal weights 1/24. Complex
    # spatial components reproduce Minkowski second/fourth contractions.
    directions=np.r_[np.eye(4),-np.eye(4),np.array(list(product([-1.,1.],repeat=4)))/2]
    directions=directions*np.array([1,1j,1j,1j])
    means=[]
    for radius in [1.,2.]:
        total=np.zeros_like(a)
        for direction in directions:
            l=_slash_many(radius*direction)
            total+=_trace_chain(vertices,[x+l for x in n])/24
        means.append(total)
    c=(means[1]-4*means[0]+3*a)/12
    b=means[0]-a-c
    return a,b,c


@lru_cache(maxsize=8)
def _simplex(order):
    z,w=quadrature(order);z=(z+1)/2;w=w/2
    u,v,t=np.meshgrid(z,z,z,indexing='ij')
    wu,wv,wt=np.meshgrid(w,w,w,indexing='ij')
    alpha=np.stack([1-u,u*(1-v),u*v*(1-t),u*v*t],axis=-1).reshape(-1,4)
    weights=(wu*wv*wt*u*u*v).ravel()
    return alpha,weights


def box_unit(mass,p1,p0,q1,q2,eps1,eps2,*,order=12,charges=(2/3,-1/3),method='wick'):
    """Finite two-source action derivative contracted with upper epsilons.

    Both photons incoming: p1=p0+q1+q2. Exact six cyclic Dirac/flavor
    orderings. Subtract zero external momenta before integration; this is
    the same removed local wave term as quantum_collective_current, NOT a
    transverse contact chosen from Ward. A conservative bound certifies
    positive Feynman denominators over the whole simplex, not only sampled
    nodes. No continuation through pair/anomalous cuts is implemented.
    """
    p1,p0,q1,q2,eps1,eps2=map(lambda x:np.asarray(x,float),(p1,p0,q1,q2,eps1,eps2))
    if any(x.shape!=(4,) for x in (p1,p0,q1,q2,eps1,eps2)) or not np.isfinite([p1,p0,q1,q2,eps1,eps2]).all():
        raise ValueError('Six finite four-vectors required.')
    if not np.isfinite(mass) or mass<=0 or not np.allclose(p1,p0+q1+q2,rtol=0,atol=1e-12):
        raise ValueError('Positive quark mass and two-photon momentum conservation required.')
    if len(charges)!=2 or not np.isfinite(charges).all(): raise ValueError('Two finite quark charges required.')
    alpha,weights=_simplex(order)
    tp=np.sqrt(2)*np.array([[0,1],[0,0]]);tm=tp.T;charge=np.diag(charges)
    fixed=(1j*G5,tm)
    other=[(1j*G5,tp,p0),(slash(eps1),charge,q1),(slash(eps2),charge,q2)]
    total=0j
    for ordering in permutations(other):
        vertices=[fixed[0]]+[x[0] for x in ordering]
        flavor=np.trace(fixed[1]@ordering[0][1]@ordering[1][1]@ordering[2][1])
        routing=np.array([np.zeros(4),ordering[0][2],ordering[0][2]+ordering[1][2],p1])
        differences=routing[:,None,:]-routing[None,:,:]
        largest=float(np.max(np.sum(differences*differences*METRIC,axis=-1)))
        # sum_{i<j} alpha_i alpha_j <=3/8 for four nonnegative weights.
        if largest>=8*mass*mass/3:
            raise ValueError('Outside the conservative certified positive-denominator domain.')
        center=alpha@routing
        shifts=routing[None,:,:]-center[:,None,:]
        delta=mass*mass-np.einsum('ni,ij,j->n',alpha,routing*routing,METRIC)+np.sum(center*center*METRIC,axis=1)
        if np.min(delta)<=0: raise ValueError('Real positive loop denominators required; cuts not implemented.')
        a,b,c=box_numerator(shifts,mass,vertices,method=method)
        a0,b0,_=box_numerator(np.zeros((4,4)),mass,vertices,method=method)
        integrand=a/delta**2-2*b/delta-6*c*np.log(delta/(mass*mass))-a0/mass**4+2*b0/mass**2
        total+=flavor*(weights@integrand)
    return float((-3/(16*np.pi**2)*total).real)


def compton_kinematics(energy,angle,mass):
    """Real photons in the initial collective particle rest frame."""
    if not np.isfinite([energy,angle,mass]).all() or min(energy,mass)<=0 or not 0<=angle<=np.pi:
        raise ValueError('Positive photon/particle energies and angle in [0,pi] required.')
    k=np.array([energy,0,0,energy]);wp=energy/(1+energy/mass*(1-np.cos(angle)))
    kp=wp*np.array([1,np.sin(angle),0,np.cos(angle)])
    p0=np.array([mass,0,0,0]);p1=p0+k-kp
    return p1,p0,k,kp


def born_unit(mass,p1,p0,q1,q2,eps1,eps2,*,order=28):
    """Tree and order-g^2 Born pieces, prescribed common on-shell pion pole.

    Sigma_R(z)=Sigma(z)-Sigma(s) fixes THIS reference pole, not QCD finite
    matching. External wave normalization is applied in compton_unit.
    Diagrams are expanded once; the loop is not Dyson-resummed.
    """
    s=dot(p0,p0)
    if abs(dot(p1,p1)-s)>1e-11: raise ValueError('Equal external virtualities required.')
    tree=0.;loop=0.
    sigma_pole=feynman_nonlocal(mass,s,'pion')
    for qa,ea,qb,eb in [(q1,eps1,q2,eps2),(q2,eps2,q1,eps1)]:
        middle=p0+qa;z=dot(middle,middle);denominator=s-z
        if abs(denominator)<1e-13: raise ValueError('Born pole excluded; use nonzero photon energy.')
        left=dot(eb,p1+middle);right=dot(ea,middle+p0)
        vleft=dot(eb,triangle_unit(mass,p1,middle,order=order))
        vright=dot(ea,triangle_unit(mass,middle,p0,order=order))
        sigma=feynman_nonlocal(mass,z,'pion')-sigma_pole
        tree+=left*right/denominator
        loop+=(vleft*right+left*vright)/denominator-left*right*sigma/denominator**2
    return dict(tree=tree,loop=loop)


def compton_unit(mass,p1,p0,q1,q2,eps1,eps2,*,order=12,triangle_order=28):
    """Charge-normalized, on-shell box+Born correction, unit g^2.

    NOT the complete action: potential-linked scalar exchange and local
    source terms are separate. This contribution need not obey alpha=-beta.
    No independent LF minus/transverse Compton calculation is supplied.
    """
    born=born_unit(mass,p1,p0,q1,q2,eps1,eps2,order=triangle_order)
    tree=2*dot(eps1,eps2)+born['tree']
    box=box_unit(mass,p1,p0,q1,q2,eps1,eps2,order=order)
    norm=bubble_vertex_norm(mass,dot(p0,p0))
    return dict(tree=tree,box=box,born=born['loop'],external_norm=-norm*tree,
                normalized_loop=box+born['loop']-norm*tree)


def scalar_photon_coefficient(mass,t,*,order=32,charges=(2/3,-1/3)):
    """a(t) in Delta L=e^2 a_i sigma_i F_em^2, unit scalar Yukawa.

    Both real photon orientations; t=(q1+q2)^2<4M^2. Independent transverse
    Dirac-loop parameter reduction: Nc Tr Q^2 M/(8pi^2) integral_triangle
    (1-4xy)/(M^2-txy). a(0)=Nc Tr Q^2/(24pi^2 M).
    """
    if not np.isfinite([mass,t]).all() or mass<=0 or t>=4*mass*mass:
        raise ValueError('Positive mass and subthreshold real-photon invariant required.')
    if len(charges)!=2 or not np.isfinite(charges).all(): raise ValueError('Two finite flavor charges required.')
    z,w=quadrature(order);z=(z+1)/2;w=w/2
    x=z[:,None];y=(1-x)*z[None,:]
    integral=np.sum(w[:,None]*w[None,:]*(1-x)*(1-4*x*y)/(mass*mass-t*x*y))
    return float(3*np.dot(charges,charges)*mass/(8*np.pi**2)*integral)


def scalar_photon_trace(mass,q1,q2,eps1,eps2,*,order=24,charges=(2/3,-1/3)):
    """Independent original scalar/two-photon Dirac triangle, both orders.

    In four dimensions the angular l^2 numerator vanishes; its evanescent
    finite local term removes the zero-momentum sigma*A^2 artifact. Subtract
    that constant, not a sigma*F^2 coefficient. This is equivalently the
    gauge-preserving finite triangle. Supports spacelike photons/transfer.
    """
    q1,q2,eps1,eps2=map(lambda x:np.asarray(x,float),(q1,q2,eps1,eps2))
    if any(x.shape!=(4,) for x in (q1,q2,eps1,eps2)) or not np.isfinite([q1,q2,eps1,eps2]).all() or not np.isfinite(mass) or mass<=0:
        raise ValueError('Positive mass and four finite photon vectors required.')
    if max(dot(q1,q1),dot(q2,q2),dot(q1+q2,q1+q2))>1e-12:
        raise ValueError('This independent trace control uses nonpositive photon/transfer invariants.')
    if len(charges)!=2 or not np.isfinite(charges).all(): raise ValueError('Two finite flavor charges required.')
    z,w=quadrature(order);z=(z+1)/2;w=w/2
    u,v=np.meshgrid(z,z,indexing='ij');wu,wv=np.meshgrid(w,w,indexing='ij')
    alpha=np.stack([1-u,u*(1-v),u*v],axis=-1).reshape(-1,3)
    weights=(wu*wv*u).ravel();total=0j
    for qa,ea,qb,eb in [(q1,eps1,q2,eps2),(q2,eps2,q1,eps1)]:
        routing=np.array([np.zeros(4),qa,qa+qb]);center=alpha@routing
        delta=mass*mass-np.einsum('ni,ij,j->n',alpha,routing*routing,METRIC)+np.sum(center*center*METRIC,axis=1)
        n=[_slash_many(routing[i]-center)+mass*np.eye(4) for i in range(3)]
        a=np.trace(n[0]@slash(ea)@n[1]@slash(eb)@n[2],axis1=-2,axis2=-1)
        a0=4*mass**3*dot(ea,eb)
        total+=weights@(a/delta-a0/mass**2)
    return float((3*np.dot(charges,charges)/(16*np.pi**2)*total).real)


def photon_vacuum_kernel(mass,reference_mass,q,eps1,eps2,*,order=64,charges=(2/3,-1/3)):
    """Two-photon determinant kernel with ONE fixed mass subtraction scale.

    The scalar source differentiates mass holding reference_mass fixed.
    Subtracting anew at each mass incorrectly erases the soft scalar F^2
    partner. This is a quark-subdiagram check, not physical charge running.
    """
    q,eps1,eps2=map(lambda x:np.asarray(x,float),(q,eps1,eps2))
    if any(x.shape!=(4,) for x in (q,eps1,eps2)) or not np.isfinite([q,eps1,eps2]).all():
        raise ValueError('Three finite photon vectors required.')
    if not np.isfinite([mass,reference_mass]).all() or min(mass,reference_mass)<=0 or dot(q,q)>1e-12:
        raise ValueError('Positive masses and spacelike photon required.')
    if len(charges)!=2 or not np.isfinite(charges).all(): raise ValueError('Two finite flavor charges required.')
    x,w=quadrature(order);x=(x+1)/2;w=w/2;b=x*(1-x)
    transverse=dot(q,q)*dot(eps1,eps2)-dot(q,eps1)*dot(q,eps2)
    return float(3*np.dot(charges,charges)/(2*np.pi**2)*np.sum(w*b*np.log((mass*mass-b*dot(q,q))/reference_mass**2))*transverse)


def scalar_exchange(model,vacuum,pion_out,pion_in,mass,t,*,order=32):
    """b_sigma(t)=a(t)*g^T(H_sigma-t)^-1*T_sigma,pi_out,pi_in, GeV^-2.

    The third derivative T=2 lambda*v*e_out*e_in belongs to the SAME
    potential. Both scalar modes and source endpoints remain. The real
    Compton contribution is -4*b_sigma*[k.k' eps.eps'-k.eps' k'.eps].
    Tree scalar response is used at this loop order, not a physical width.
    """
    return float(np.asarray(pion_out)@scalar_exchange_matrix(model,vacuum,mass,t,order=order)@np.asarray(pion_in))


def scalar_exchange_matrix(model,vacuum,mass,t,*,order=32):
    """Open external pion multiplet indices; NOT the rank-one quark box.

    T_{aij}=2 lambda_a v_a delta_ai delta_aj in the original local-potential
    coordinates. Under a basis change transport the FULL third derivative,
    Hessian and source, not only the quartic parameter list.
    """
    _,h=model.matrices(vacuum)
    a=scalar_photon_coefficient(mass,t,order=order)
    return a*np.diag(2*model.quartic*np.asarray(vacuum)*np.linalg.solve(h-t*np.eye(2),model.yukawa))


def dipole_estimate(mass,external_mass,energy,*,order=12,yukawa_squared=1.):
    """Finite-energy estimates, not exact zero-energy polarizabilities.

    Forward/backward perpendicular real polarizations give alpha+beta and
    alpha-beta. Extrapolate energy explicitly; no alpha=-beta assumption.
    Output fm^3; scalar exchange/local terms are NOT included here.
    """
    if not np.isfinite(yukawa_squared) or yukawa_squared<0: raise ValueError('Nonnegative Yukawa squared required.')
    eps=np.array([0.,0,1,0]);values=[]
    for angle in [0.,np.pi]:
        p1,p0,k,kp=compton_kinematics(energy,angle,external_mass)
        # Rest-frame transverse polarizations kill BOTH Born products.
        # The general Born implementation remains independently tested.
        loop=box_unit(mass,p1,p0,k,-kp,eps,eps,order=order)+2*bubble_vertex_norm(mass,external_mass**2)
        values.append(yukawa_squared*ALPHA_EM*HBARC**3/(2*external_mass)*loop/(k[0]*kp[0]))
    return dict(alpha_plus_beta=values[0],alpha_minus_beta=values[1],
                alpha=(values[0]+values[1])/2,beta=(values[0]-values[1])/2)


def absorption_kinematics(mass,external_mass,s,cosine):
    """gamma + prescribed charged collective -> u + dbar, pair CM frame."""
    cosine=np.asarray(cosine,float)
    if not np.isfinite([mass,external_mass,s]).all() or not 0<external_mass<2*mass or s<=4*mass*mass or not np.isfinite(cosine).all() or np.max(abs(cosine))>1:
        raise ValueError('Subthreshold positive external mass, above-threshold s and physical angles required.')
    root=np.sqrt(s);w=(s-external_mass**2)/(2*root);p=np.array([root-w,0,0,-w]);k=np.array([w,0,0,w])
    length=np.sqrt(s/4-mass*mass)
    quark=np.stack([np.full_like(cosine,root/2),length*np.sqrt(1-cosine*cosine),np.zeros_like(cosine),length*cosine],axis=-1)
    antiquark=quark*np.array([1,-1,-1,-1])
    return p,k,quark,antiquark


def absorption_parts(mass,external_mass,s,cosine,epsilon,*,charges=(2/3,-1/3)):
    """Three tree source attachments before external ubar/v contraction.

    Each is a Dirac matrix; sqrt(2) pion flavor matrix, Nc colors and electric
    charge are supplied when taking the cross section. Keeping the scalar
    charged-pion attachment is necessary for Ward, although it vanishes in
    the CM transverse gauge. This is an unconfined reference-quark channel.
    """
    p,k,q,a=absorption_kinematics(mass,external_mass,s,cosine)
    eps=np.asarray(epsilon,float)
    if eps.shape!=(4,) or not np.isfinite(eps).all() or len(charges)!=2 or not np.isfinite(charges).all():
        raise ValueError('Finite photon polarization and two flavor charges required.')
    nq=_slash_many(q-k)+mass*np.eye(4);na=_slash_many(k-a)+mass*np.eye(4)
    dq=np.sum((q-k)**2*METRIC,axis=-1)-mass*mass
    da=np.sum((k-a)**2*METRIC,axis=-1)-mass*mass
    first=charges[0]*(slash(eps)@nq@(1j*G5))/dq[...,None,None]
    second=charges[1]*((1j*G5)@na@slash(eps))/da[...,None,None]
    collective=(charges[0]-charges[1])*dot(2*p+k,eps)/(s-external_mass**2)*(1j*G5)
    return dict(quark=first,antiquark=second,collective=collective,total=first+second+collective)


def absorption_cross_section(mass,external_mass,s,*,order=64,yukawa_squared=1.):
    """Independent tree cross section, GeV^-2, actual e^2 restored.

    sigma=beta/[32pi(s-m_pi^2)] int_-1^1 dc (1/2) sum_pol,spin,color |M|^2.
    Both final spins, Nc=3 and the charged flavor factor 2 remain; no photon
    helicity sum is confused with its initial-state average. No fit inputs.
    """
    if not np.isfinite(yukawa_squared) or yukawa_squared<0: raise ValueError('Nonnegative Yukawa squared required.')
    x,w=quadrature(order);p,k,q,a=absorption_kinematics(mass,external_mass,s,x)
    qsum=_slash_many(q)+mass*np.eye(4);asum=_slash_many(a)-mass*np.eye(4)
    result=0.
    for eps in [np.array([0.,1,0,0]),np.array([0.,0,1,0])]:
        vertex=absorption_parts(mass,external_mass,s,x,eps)['total']
        bar=GAMMA[0]@vertex.conj().swapaxes(-1,-2)@GAMMA[0]
        norm=np.trace(qsum@vertex@asum@bar,axis1=-2,axis2=-1).real
        result+=.5*(w@norm)*2*3*(4*np.pi*ALPHA_EM)*yukawa_squared
    return float(np.sqrt(1-4*mass*mass/s)/(32*np.pi*(s-external_mass**2))*result)


def absorption_dipole_sum(mass,external_mass,*,order=64,angular_order=96,yukawa_squared=1.):
    """Independent forward dispersion (Baldin) integral, fm^3.

    alpha+beta=1/(2pi^2) int_{omega_th}^infinity sigma(omega)/omega^2 domega.
    s=m_pi^2+2m_pi*omega; integrate s=4M^2/(1-b^2), 0<b<1. This validates
    the free-quark reference loop, NOT a physical hadronic absorption sum.
    """
    b,w=quadrature(order);b=(b+1)/2;w=w/2
    s=4*mass*mass/(1-b*b);jac=8*mass*mass*b/(1-b*b)**2
    sigma=np.array([absorption_cross_section(mass,external_mass,z,order=angular_order,yukawa_squared=yukawa_squared) for z in s])
    return float(HBARC**3*external_mass/np.pi**2*np.sum(w*jac*sigma/(s-external_mass**2)**2))
