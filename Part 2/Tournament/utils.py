#copied from CV RL course
import logging
from PIL import Image, ImageDraw, ImageFont
import moviepy.editor as mp
import random

def create_gif(images_lst, agent_left_lbl, agent_right_lbl, match_number, VIDEOS_PATH, MATCH_NAME):
    images = []

    for img in images_lst:
        # Convert the rendered image to a PIL Image for annotation
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil) 
        
        # Add text to the image
        font = ImageFont.load_default() 
        draw.text((50, 23), f"Match {match_number} / 5", fill="black", font=font) # number of match
        draw.text((10, 195), f"Group {agent_left_lbl}", fill="black", font=font) # left
        draw.text((90, 195), f"Group {agent_right_lbl}", fill="black", font=font) # right
        images.append(img_pil)

    # Save as GIF
    output = VIDEOS_PATH + MATCH_NAME +".gif"
    images[0].save(output, save_all=True, append_images=images[1:], duration=10, loop=0)
    print("Exporting GIF at {}".format(output))


def create_mp4(agent_left_lbl, agent_right_lbl, match_number, VIDEOS_PATH, MATCH_NAME):
    output = VIDEOS_PATH + MATCH_NAME

    # loading GIF file
    clip = mp.VideoFileClip(output +".gif")
    # converting to MP4
    clip.write_videofile(output +".mp4")
    print("Exporting video at {}".format(output +".mp4"))


def set_logging(logging_level=logging.DEBUG, logging_name="log"):
        # remove older loggers
        log = logging.getLogger()  # root logger
        for hdlr in log.handlers[:]:  # remove all old handlers
            log.removeHandler(hdlr)

        # Start logging system
        logging.basicConfig(filename="./"+ logging_name +".txt", format="%(asctime)s %(levelname)s %(message)s", filemode='w', level=logging_level)
        logging.getLogger('matplotlib.font_manager').disabled = True

        return log
