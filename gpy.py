import numpy as np
from scipy import linalg
from scipy import optimize
from scipy.linalg import lapack, blas
import GPy

'''
Contains functions from GPy as they are
'''

def jitchol(A, maxtries=5):
    A = np.ascontiguousarray(A)
    L, info = lapack.dpotrf(A, lower=1)
    if info == 0:
        return L
    else:
        diagA = np.diag(A)
        if np.any(diagA <= 0.):
            raise linalg.LinAlgError("not pd: non-positive diagonal elements")
        jitter = diagA.mean() * 1e-6
        num_tries = 1
        while num_tries <= maxtries and np.isfinite(jitter):
            try:
                L = linalg.cholesky(A + np.eye(A.shape[0]) * jitter, lower=True)
                return L
            except:
                jitter *= 10
            finally:
                num_tries += 1
        raise linalg.LinAlgError("not positive definite, even with jitter.")

    return L
    
def dtrtrs(A, B, lower=1, trans=0, unitdiag=0):
    """
    Wrapper for lapack dtrtrs function

    DTRTRS solves a triangular system of the form

        A * X = B  or  A**T * X = B,

    where A is a triangular matrix of order N, and B is an N-by-NRHS
    matrix.  A check is made to verify that A is nonsingular.

    :param A: Matrix A(triangular)
    :param B: Matrix B
    :param lower: is matrix lower (true) or upper (false)
    :returns: Solution to A * X = B or A**T * X = B

    """
    A = np.asfortranarray(A)
    #Note: B does not seem to need to be F ordered!
    return lapack.dtrtrs(A, B, lower=lower, trans=trans, unitdiag=unitdiag)
    
def compute_B_statistics(K, W, log_concave, *args, **kwargs):
    if not log_concave:
        #print "Under 1e-10: {}".format(np.sum(W < 1e-6))
        W = np.clip(W, 1e-6, 1e+30)
        # For student-T we can clip this more intelligently. If the
        # objective has hardly changed, we can increase the clipping limit
        # by ((v+1)/v)/sigma2
        # NOTE: when setting a parameter inside parameters_changed it will allways come to closed update circles!!!
        #W.__setitem__(W < 1e-6, 1e-6, update=False)  # FIXME-HACK: This is a hack since GPy can't handle negative variances which can occur
                            # If the likelihood is non-log-concave. We wan't to say that there is a negative variance
                            # To cause the posterior to become less certain than the prior and likelihood,
                            # This is a property only held by non-log-concave likelihoods
    if np.any(np.isnan(W)):
        raise ValueError('One or more element(s) of W is NaN')
    #W is diagonal so its sqrt is just the sqrt of the diagonal elements
    W_12 = np.sqrt(W)
    K = np.array(K)
    B = np.eye(K.shape[0]) + W_12*K*W_12.T
    L = jitchol(B)

    LiW12, _ = dtrtrs(L, np.diagflat(W_12), lower=1, trans=0)
    K_Wi_i = np.dot(LiW12.T, LiW12) # R = W12BiW12, in R&W p 126, eq 5.25

    #here's a better way to compute the required matrix.
    # you could do the model finding witha backsub, instead of a dot...
    #L2 = L/W_12
    #K_Wi_i_2 , _= dpotri(L2)
    #symmetrify(K_Wi_i_2)

    #compute vital matrices
    C = np.dot(LiW12, K)
    Ki_W_i = K - C.T.dot(C)

    I_KW_i = np.eye(K.shape[0]) - np.dot(K, K_Wi_i)
    logdet_I_KW = 2*np.sum(np.log(np.diag(L)))

    return K_Wi_i, logdet_I_KW, I_KW_i, Ki_W_i
    
_lim_val = np.finfo(np.float32).max
_lim_val_square = np.sqrt(_lim_val)

def chain_1(df_dg, dg_dx):
    if np.all(dg_dx==1.):
        return df_dg
    return df_dg * dg_dx

def chain_2(d2f_dg2, dg_dx, df_dg, d2g_dx2):
    if np.all(dg_dx==1.) and np.all(d2g_dx2 == 0):
        return d2f_dg2
    dg_dx_2 = np.clip(dg_dx, -np.inf, _lim_val_square)**2
    #dg_dx_2 = dg_dx**2
    return d2f_dg2*(dg_dx_2) + df_dg*d2g_dx2

# that's for Bernoulli likelihood only
def dlogpdf_dlink_(inv_link_f, y):
    #grad = (y/inv_link_f) - (1.-y)/(1-inv_link_f)
    #grad = np.where(y, 1./inv_link_f, -1./(1-inv_link_f))
    ff = np.clip(inv_link_f, 1e-7, 1-1e-7)
    denom = np.where(y==1, ff, -(1-ff))
    return 1./denom

def d2logpdf_dlink2_(inv_link_f, y):
    #d2logpdf_dlink2 = -y/(inv_link_f**2) - (1-y)/((1-inv_link_f)**2)
    #d2logpdf_dlink2 = np.where(y, -1./np.square(inv_link_f), -1./np.square(1.-inv_link_f))
    arg = np.where(y==1, inv_link_f, 1.-inv_link_f)
    ret =  -1./np.square(np.clip(arg, 1e-7, 1e9))
    if np.any(np.isinf(ret)):
        stop
    return ret

_sqrt_2pi = np.sqrt(2*np.pi)
def std_norm_pdf(x):
    x = np.clip(x,-1e100,1e100)
    return np.exp(-np.square(x)/2)/_sqrt_2pi

def dlogpdf_df(f, y, lik):
    if isinstance(lik.gp_link, GPy.likelihoods.link_functions.Identity):
        return dlogpdf_dlink_(f, y)
    elif isinstance(lik.gp_link, GPy.likelihoods.link_functions.Probit):
        inv_link_f = lik.gp_link.transf(f)
        dlogpdf_dlink = dlogpdf_dlink_(inv_link_f, y)
        
        dlink_df = std_norm_pdf(f)
        return chain_1(dlogpdf_dlink, dlink_df)
    
def d2logpdf_df2(f, y, lik):
    if isinstance(lik.gp_link, GPy.likelihoods.link_functions.Identity):
        return d2logpdf_dlink2_(f, y)
    else:
        inv_link_f = lik.gp_link.transf(f)
        d2logpdf_dlink2 = d2logpdf_dlink2_(inv_link_f, y)
        dlink_df = lik.gp_link.dtransf_df(f)
        dlogpdf_dlink = dlogpdf_dlink_(inv_link_f, y)
        d2link_df2 = lik.gp_link.d2transf_df2(f)
        return chain_2(d2logpdf_dlink2, dlink_df, dlogpdf_dlink, d2link_df2)

def logpdf_link(inv_link_f, y):
    #objective = y*np.log(inv_link_f) + (1.-y)*np.log(inv_link_f)
    p = np.where(y==1, inv_link_f, 1.-inv_link_f)
    return np.log(np.clip(p, 1e-7 ,np.inf))
    
def logpdf(f, y, lik):
    if isinstance(lik.gp_link, GPy.likelihoods.link_functions.Identity):
        return logpdf_link(f, y)
    else:
        inv_link_f = lik.gp_link.transf(f)
        return logpdf_link(inv_link_f, y)
    
def logPdfNormal(z):
    """
    Robust implementations of log pdf of a standard normal.

     @see [[https://github.com/mseeger/apbsint/blob/master/src/eptools/potentials/SpecfunServices.h original implementation]]
     in C from Matthias Seeger.
    """
    return -0.5 * (M_LN2PI + z * z)
    
def _erfRationalHelper(x):
    assert x > 0.0, "Arg of erfRationalHelper should be >0.0; was {}".format(x)

    if (x >= ERF_CODY_LIMIT2):
        """
         x/sqrt(2) >= 4

         Q(x)   = 1 + sqrt(pi) y R_1(y),
         R_1(y) = poly(p_j,y) / poly(q_j,y),  where  y = 2/(x*x)

         Ordering of arrays: 4,3,2,1,0,5 (only for numerator p_j; q_5=1)
         ATTENTION: The p_j are negative of the entries here
         p (see P1_ERF)
         q (see Q1_ERF)
        """
        y = 2.0 / (x * x)

        res = y * P1_ERF[5]
        den = y
        i = 0

        while (i <= 3):
            res = (res + P1_ERF[i]) * y
            den = (den + Q1_ERF[i]) * y
            i += 1

        # Minus, because p(j) values have to be negated
        return 1.0 - M_SQRTPI * y * (res + P1_ERF[4]) / (den + Q1_ERF[4])
    else:
        """
         x/sqrt(2) < 4, x/sqrt(2) >= 0.469

         Q(x)   = sqrt(pi) y R_2(y),
         R_2(y) = poly(p_j,y) / poly(q_j,y),   y = x/sqrt(2)

         Ordering of arrays: 7,6,5,4,3,2,1,0,8 (only p_8; q_8=1)
         p (see P2_ERF)
         q (see Q2_ERF
        """
        y = x / M_SQRT2
        res = y * P2_ERF[8]
        den = y
        i = 0

        while (i <= 6):
            res = (res + P2_ERF[i]) * y
            den = (den + Q2_ERF[i]) * y
            i += 1

        return M_SQRTPI * y * (res + P2_ERF[7]) / (den + Q2_ERF[7])

def _erfRationalHelperR3(y):
    assert y >= 0.0, "Arg of erfRationalHelperR3 should be >=0.0; was {}".format(y)

    nom = y * P3_ERF[4]
    den = y
    i = 0
    while (i <= 2):
        nom = (nom + P3_ERF[i]) * y
        den = (den + Q3_ERF[i]) * y
        i += 1
    return (nom + P3_ERF[3]) / (den + Q3_ERF[3])

ERF_CODY_LIMIT1 = 0.6629
ERF_CODY_LIMIT2 = 5.6569
M_LN2PI         = 1.83787706640934533908193770913
M_LN2           = 0.69314718055994530941723212146
M_SQRTPI        = 1.77245385090551602729816748334
M_SQRT2         = 1.41421356237309504880168872421

#weights for the erfHelpers (defined here to avoid redefinitions at every call)
P1_ERF = [
3.05326634961232344e-1, 3.60344899949804439e-1,
1.25781726111229246e-1, 1.60837851487422766e-2,
6.58749161529837803e-4, 1.63153871373020978e-2]
Q1_ERF = [
2.56852019228982242e+0, 1.87295284992346047e+0,
5.27905102951428412e-1, 6.05183413124413191e-2,
2.33520497626869185e-3]
P2_ERF = [
5.64188496988670089e-1, 8.88314979438837594e+0,
6.61191906371416295e+1, 2.98635138197400131e+2,
8.81952221241769090e+2, 1.71204761263407058e+3,
2.05107837782607147e+3, 1.23033935479799725e+3,
2.15311535474403846e-8]
Q2_ERF = [
1.57449261107098347e+1, 1.17693950891312499e+2,
5.37181101862009858e+2, 1.62138957456669019e+3,
3.29079923573345963e+3, 4.36261909014324716e+3,
3.43936767414372164e+3, 1.23033935480374942e+3]
P3_ERF = [
3.16112374387056560e+0, 1.13864154151050156e+2,
3.77485237685302021e+2, 3.20937758913846947e+3,
1.85777706184603153e-1]
Q3_ERF = [
2.36012909523441209e+1, 2.44024637934444173e+2,
1.28261652607737228e+3, 2.84423683343917062e+3]

def std_norm_cdf(z):
    for i in range (len(z)):
        z[i] = std_norm_cdf_num(z[i])
    return z

def std_norm_cdf_num(z):
    if (abs(z) < ERF_CODY_LIMIT1):
        # Phi(z) approx (1+y R_3(y^2))/2, y=z/sqrt(2)
        return 0.5 * (1.0 + (z / M_SQRT2) * _erfRationalHelperR3(0.5 * z * z))
    elif (z < 0.0):
        # Phi(z) approx N(z)Q(-z)/(-z), z<0
        return np.exp(logPdfNormal(z)) * _erfRationalHelper(-z) / (-z)
    else:
        return 1.0 - np.exp(logPdfNormal(z)) * _erfRationalHelper(z) / z