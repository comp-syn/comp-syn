#!/usr/bin/env python3

from compsyn.wordtocolor_vector import WordToColorVector

def test_w2cv_fresh_run():
    w2cv = WordToColorVector(label="dog")
    w2cv.run_image_capture()
    w2cv.run_analysis()
    import IPython; IPython.embed()

if __name__ == "__main__":
    test_w2cv_fresh_run()
