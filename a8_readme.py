def time_spent():
  '''N: # of hours you spent on this one'''
  return 7

def collaborators():
  '''Eg. ppl=['batman', 'ninja'] (use their athena username)'''
  return ['rlacey']

def potential_issues():
  return 'The Poisson without CG seems to converge a bit slowly (in ~2500 iterations the bear definitely looked good). Others seemed to have this issue too. The Pru was small, so sometimes it was hard to tell whether things were good or not, but I think everything should work.'

def extra_credit():
#```` Return the function names you implemended````
#```` Eg. return ['full_sift', 'bundle_adjustment']````
  return []

def most_exciting():
  return 'Getting the Poisson solvers to work -- the composited images look nice'

def most_difficult():
  return 'Figuring out how to handle noise. Also getting PNGs to work for my own composite at the end'

def my_composition():
  input_images=['bg.png', 'dolphin.png', 'dolphin_mask.png']
  output_images='myowncomposite.png'
  return (input_images, output_images)

def my_debug():
  '''return (1) a string explaining how you debug
  (2) the images you used in debugging.

  Eg
  images=['debug1.jpg', 'debug2jpg']
  my_debug='I used blahblahblah...
  '''
  my_debug = 'I used the provided images of the pru. In addition, I did a fair amount of math by hand (for example, deriving what M transpose was in terms of the kernel, I made a small non-symmetric kernel, applied it to a column vector to get the matrix, and looked at the kernel that gave M^T. Also, played a lot with the parameters and printed energies to make sure they were decreasing.'

  images = ['pru.png', 'bear.png']
  return (my_debug, images)
