import numpy as np

# === Deconvolution with gradient descent ===

def dotIm(im1, im2):
  return np.sum(im1 * im2)

def applyKernel(im, kernel):
  ''' return Mx, where x is im '''
  return convolve3(im, kernel)

def applyConjugatedKernel(im, kernel):
  ''' return M^T x, where x is im '''
  return convolve3(im, np.fliplr(np.flipud(kernel)));

def applyAMatrix(im, kernel):
  ''' return Ax, where A = M^TM'''
  return applyConjugatedKernel(applyKernel(im, kernel), kernel)

def computeResidual(kernel, x, y):
  ''' return MTy - MTMx '''
  return applyConjugatedKernel(y, kernel) - applyAMatrix(x, kernel)

def computeStepSize(r, kernel):
  return dotIm(r, r) / dotIm(r, applyAMatrix(r, kernel))

def deconvGradDescent(im_blur, kernel, niter=10):
  ''' return deblurred image '''
  im = np.zeros(im_blur.shape)
  for n in xrange(niter):
    r = computeResidual(kernel, im, im_blur)
    im += computeStepSize(r, kernel) * r
  return im

# === Deconvolution with conjugate gradient ===

def computeGradientStepSize(r, d, kernel):
  return dotIm(r, r) / dotIm(d, applyAMatrix(d, kernel))

def computeConjugateDirectionStepSize(old_r, new_r):
  return dotIm(new_r, new_r) / dotIm(old_r, old_r)

def deconvCG(im_blur, kernel, niter=10):
  ''' return deblurred image '''
  im = np.zeros(im_blur.shape)
  r0 = computeResidual(kernel, im, im_blur)
  d = r0.copy()
  for n in xrange(niter):
    alpha = computeGradientStepSize(r0, d, kernel)
    im += alpha * d
    r1 = r0 - alpha * applyAMatrix(d, kernel)
    d = r1 + computeConjugateDirectionStepSize(r0, r1) * d
    r0 = r1.copy()
  return im

def laplacianKernel():
  ''' a 3-by-3 array '''
  return np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])

def applyLaplacian(im):
  ''' return Lx (x is im)'''
  return applyKernel(im, laplacianKernel())

def applyRegularizedOperator(im, kernel, lamb):
  ''' (A + lambda L )x'''
  return applyAMatrix(im, kernel) + lamb * applyLaplacian(im)

def computeGradientStepSize_reg(grad, p, kernel, lamb):
  return dotIm(grad, grad) / dotIm(p, applyRegularizedOperator(p, kernel, lamb))

def deconvCG_reg(im_blur, kernel, lamb=0.05, niter=10):
  ''' return deblurred and regularized im '''
  im = np.zeros(im_blur.shape)
  r0 = applyConjugatedKernel(im_blur, kernel) - applyRegularizedOperator(im, kernel, lamb)
  d = r0.copy()
  for n in xrange(niter):
    alpha = computeGradientStepSize_reg(r0, d, kernel, lamb)
    im += alpha * d
    r1 = r0 - alpha * applyRegularizedOperator(d, kernel, lamb)
    d = r1 + computeConjugateDirectionStepSize(r0, r1) * d
    r0 = r1.copy()
  return im

def naiveComposite(bg, fg, mask, y, x):
  ''' naive composition'''
  height, width, c = fg.shape
  out = bg.copy()
  out[y:y+height, x:x+width] = np.where(mask, fg*mask, bg[y:y+height, x:x+width])
  return out

def Poisson(bg, fg, mask, niter=3000):
  ''' Poisson editing using gradient descent'''
  b = applyLaplacian(fg)
  x = np.where(mask, 0.0, bg)
  for n in xrange(niter):
    r = (b - applyLaplacian(x)) * mask
    alpha = dotIm(r, r) / dotIm(r, applyLaplacian(r))
    x += alpha * r
  return x

def PoissonCG(bg, fg, mask, niter=150):
  ''' Poison editing using conjugate gradient '''
  b = applyLaplacian(fg)
  x = np.where(mask, 0.0, bg)
  r0 = (b - applyLaplacian(x)) * mask
  d = r0.copy()
  for n in xrange(niter):
    alpha = dotIm(r0, r0) / dotIm(d, applyLaplacian(d))
    x += alpha * d
    r1 = (r0 - alpha * applyLaplacian(d)) * mask
    d = r1 + computeConjugateDirectionStepSize(r0, r1) * d
    r0 = r1.copy()
  return x

#==== Helpers. Use them as possible. ====

def convolve3(im, kernel):
  from scipy import ndimage
  center=(0,0)
  r=ndimage.filters.convolve(im[:,:,0], kernel, mode='reflect', origin=center)
  g=ndimage.filters.convolve(im[:,:,1], kernel, mode='reflect', origin=center)
  b=ndimage.filters.convolve(im[:,:,2], kernel, mode='reflect', origin=center)
  return (np.dstack([r,g,b]))

def gauss2D(sigma=2, truncate=3):
  kernel=horiGaussKernel(sigma, truncate);
  kerker=np.dot(kernel.transpose(), kernel)
  return kerker/sum(kerker.flatten())

def horiGaussKernel(sigma, truncate=3):
  from scipy import signal
  sig=signal.gaussian(2*int(sigma*truncate)+1,sigma)
  return np.array([sig/sum(sig)])



