import sys
import segmentation
from moviepy.editor import VideoFileClip


def process_video(video_file):
    model = segmentation.load_seg_model("car_model.h5")

    def process_full(img_file):
        feature_map = segmentation.seg_by_model(img_file, model)
        return segmentation.grid_all(img_file, feature_map)
    
    clip = VideoFileClip(video_file)
    return clip.fl_image(process_full) #NOTE: this function expects color images!!

if __name__ == '__main__':
    assert (len(sys.argv) == 3), 'Need input and output video file names.'
    white_output = sys.argv[2]
    white_clip = process_video(sys.argv[1])
    white_clip.write_videofile(white_output, audio=False)

