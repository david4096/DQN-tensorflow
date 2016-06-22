import time
import cPickle
import tensorflow as tf

def timeit(f):
  def timed(*args, **kwargs):
    start_time = time.time()
    result = f(*args, **kwargs)
    end_time = time.time()

    print("   [-] %s : %2.5f sec" % (f.__name__, end_time - start_time))
    return result
  return timed

def get_time():
  return time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())

def calc_gpu_fraction(fraction_string):
  idx, num = fraction_string.split('/')
  idx, num = float(idx), float(num)

  fraction = 1 / (num - idx + 1)
  print " [*] GPU : %.4f" % fraction
  return fraction
