import call_sfm
import numpy as np

def main():
    i = "/home/matteo/SFM/experiments/results/matches/sfm_data.json"
    m = "/home/matteo/SFM/experiments/results/matches"
    o = "/home/matteo/SFM/experiments/results/reconstruction"
    o_2 = "/home/matteo/SFM/experiments/results/reconstruction_test"
    col = call_sfm.IntVector(500000)
    col2 = call_sfm.IntVector(500000)
    row = call_sfm.IntVector(500000)
    row2 = call_sfm.IntVector(500000)
    grad = call_sfm.DoubleVector(500000)
    grad2 = call_sfm.DoubleVector(500000)
    o1 = call_sfm.DoubleVector()
    o2 = call_sfm.DoubleVector()
    pc = call_sfm.DoubleVector()
    we = call_sfm.DoubleVector()

    call_sfm.phase_one(i, m, o, col, row, grad)
    print ("part one is working! part one is working! part one is working! part one is working! part one is working!")
    call_sfm.final_ba(i, m, o_2, col2, row2, grad2)
    print ("part two is working! part two is working! part two is working! part two is working! part two is working!")
    call_sfm.call_sfm(i, o1, o2, pc)
    print ("part three is working! part three is working! part three is working! part three is working! part three is working!")

if __name__ == "__main__":
    main()
