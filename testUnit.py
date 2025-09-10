
import numpy as np
import time
from myMath import myMath

def chrono(func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    print(f"Execution time: {end - start:.6f} seconds")

def main():
    z_single_p = 1.
    z_single_n = -1.
    z_list = [2., 0., -1.]
    z_listbig = np.linspace(-5, 150, 500).astype(float)

    # print("ACTIVATION FUNCTION")


    # print("sigmoid //")
    # print(f"sigmoid {chrono(myMath.sigmoid, z_single_n)}")
    # print(f"sigmoid {chrono(myMath.sigmoid, z_single_p)}")
    # print(f"sigmoid {chrono(myMath.sigmoid, z_list)}")

    # print("sigmoid Prime //")
    # print(f"sigmoid Prime {chrono(myMath.sigmoidPrime, z_single_n)}")
    # print(f"sigmoid Prime {chrono(myMath.sigmoidPrime, z_single_p)}")
    # print(f"sigmoid Prime {chrono(myMath.sigmoidPrime, z_list)}")
    
    # print("reLu //")
    # print(f"reLu {chrono(myMath.reLu, z_single_n)}")
    # print(f"reLu {chrono(myMath.reLu, z_single_p)}")
    # print(f"reLu {chrono(myMath.reLu, z_list)}")

    # print("sigmoid Prime //")
    # print(f"reLu Prime {chrono(myMath.reLuPrime, z_single_n)}")
    # print(f"reLu Prime {chrono(myMath.reLuPrime, z_single_p)}")
    # print(f"reLu Prime {chrono(myMath.reLuPrime, z_list)}")

    # print("leaky ReLu //")
    # print(f"leaky ReLu {chrono(myMath.leakyReLu, z_single_n)}")
    # print(f"leaky ReLu {chrono(myMath.leakyReLu, z_single_p)}")
    # print(f"leaky ReLu {chrono(myMath.leakyReLu, z_list)}")

    # print("leaky ReLu Prime //")
    # print(f"leaky ReLu Prime {chrono(myMath.leakyReluPrime, z_single_n)}")
    # print(f"leaky ReLu Prime {chrono(myMath.leakyReluPrime, z_single_p)}")
    # print(f"leaky ReLu Prime {chrono(myMath.leakyReluPrime, z_list)}")

    # print("Tanh //")
    # print(f"Tanh {chrono(myMath.tanh, z_single_n)}")
    # print(f"Tanh {chrono(myMath.tanh, z_single_p)}")
    # print(f"Tanh {chrono(myMath.tanh, z_list)}")

    # print("Tanh Prime //")
    # print(f"Tanh Prime {chrono(myMath.tanhPrime, z_single_n)}")
    # print(f"Tanh Prime {chrono(myMath.tanhPrime, z_single_p)}")
    # print(f"Tanh Prime {chrono(myMath.tanhPrime, z_list)}")

    # print("Step //")
    # print(f"Step {chrono(myMath.step, z_single_n)}")
    # print(f"Step {chrono(myMath.step, z_single_p)}")
    # print(f"Step {chrono(myMath.step, z_list)}")

    # print("Step Prime //")
    # print(f"Step Prime {chrono(myMath.stepPrime, z_single_n)}")
    # print(f"Step Prime {chrono(myMath.stepPrime, z_single_p)}")
    # print(f"Step Prime {chrono(myMath.stepPrime, z_list)}")
    # print()

    # print("RANDOM INITIALIZER")

    # print("random normal //")
    # print(myMath.randomNormal((0,0)))
    # print(myMath.randomNormal((1,1)))
    # print(myMath.randomNormal((2,2)))
    # print(myMath.randomNormal())

    # print("random uniform //")
    # print(myMath.randomUniform((0,0)))
    # print(myMath.randomUniform((1,1)))
    # print(myMath.randomUniform((2,2)))
    # print(myMath.randomUniform())

    # print("zeros //")
    # print(myMath.zeros((0,0)))
    # print(myMath.zeros((1,1)))
    # print(myMath.zeros((2,2)))
    # print(myMath.zeros())

    # print("ones //")
    # print(myMath.ones((0,0)))
    # print(myMath.ones((1,1)))
    # print(myMath.ones((2,2)))
    # print(myMath.ones())

    # print("xavier normal //")
    # print(myMath.xavierNormal((0,0)))
    # print(myMath.xavierNormal((1,1)))
    # print(myMath.xavierNormal((2,2)))
    # print(myMath.xavierNormal())

    # print("xavier Uniform //")
    # print(myMath.xavierUniform((0,0)))
    # print(myMath.xavierUniform((1,1)))
    # print(myMath.xavierUniform((2,2)))
    # print(myMath.xavierUniform())

    # print("heNormal //")
    # print(myMath.heNormal((0,0)))
    # print(myMath.heNormal((1,1)))
    # print(myMath.heNormal((2,2)))
    # print(myMath.heNormal())

    # print("heUniform //")
    # print(myMath.heUniform((0,0)))
    # print(myMath.heUniform((1,1)))
    # print(myMath.heUniform((2,2)))
    # print(myMath.heUniform())




if __name__ == "__main__":
    main()