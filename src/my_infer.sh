
# infer
python ./src/my_multibee_imgdemo_withangle.py ctdet --fix_res --demo ./images_gray \
--K 2000 --load_model /home/alex/works/Centernet/CenterNet/exp/ctdet/coco_multibee_withangle/model_best.pth \
--debug 4 --keep_res
# python ./src/my_multibee_imgdemo.py ctdet --input_res 2048 --demo ./images \
# --K 2000 --load_model /home/alex/works/Centernet/CenterNet/exp/ctdet/coco_multibee/model_best.pth \
# --debug 4 --keep_res

# python ./src/my_multibee_imgdemo_gray.py ctdet --fix_res --demo ./images_gray \
# --K 2000 --load_model /home/alex/works/Centernet/CenterNet/exp/ctdet/coco_multibee_gray/model_best.pth \
# --debug 4 --keep_res

#python ./src/my_multibee_imgdemo_withbackground.py ctdet --input_res 2048 --demo ./images \
#--K 2000 --load_model /home/alex/works/Centernet/CenterNet/exp/ctdet/coco_multibee_withbackground/model_best.pth \
#--debug 4 --keep_res