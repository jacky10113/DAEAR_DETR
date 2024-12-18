## 1.Install
### 1.1Create a virtual environment
conda create -n daeardetr python=3.8 -y
conda activate daeardetr
### 1.2Install dependencies
pip install -r requirements.txt

## 2.Train
_training on single-gpu_
'python tools/train.py -c configs/daear_detr/daear_detr_r50vd_6x_caltechPedestrian_dualencode_level4_encodestage2_residual_AttentionGated.yml'

## 3.Test
'python tools/train.py -c configs/daear_detr/daear_detr_r50vd_6x_caltechPedestrian_dualencode_level4_encodestage2_residual_AttentionGated.yml -r ../models_trained/object_detection/daear_detr/caltechPedestrian/daear_detr_dualencode_level4_encodestage2_residual_AttentionGated_r50vd_6x_caltechPedestrian/20240920_184030/checkpoint0063.pth  --test-only'

_Executing the above command will generate prediction data for the test set, the file name is predictions.json_

## 4.Calculate LAMR
### 4.1  
_Modify result_jsons in tools/convert_json_to_txt.py to the real path of predictions.json_
### 4.2  
'python tools/convert_json_to_txt.py'
### 4.3  
_Copy the txt format data generated in the ./test_out/eval_caltech/ directory to caltech_tool to calculate LAMR and draw the corresponding curve_

## 5.Download links of  trained models and prediction  on Caltech Pedestrian Dataset(interval 30)
### 5.1 DAEAR-DETR  
- model https://drive.google.com/file/d/14uq3xefOhYFStYO0Hsni7xQoZ-G31S8-/view?usp=sharing
- predictions(json) https://drive.google.com/file/d/1thE7GqkTH1whf703Qcc5rMbpMpWEgrvK/view?usp=drive_link
- predictions(txt) https://drive.google.com/drive/folders/1xBex_cNNrLJxWXeZ2sPLhdLD9Yjk3NrR?usp=sharing

### 5.2 Faster-RCNN
model https://drive.google.com/file/d/1TZa6n8PwRCl-vKxvCB5UQ_5KtonbziFc/view?usp=drive_link
predictions(json) https://drive.google.com/file/d/1OFh6Uovl67lwbdD-HObbNsKx5x2OKOXB/view?usp=drive_link
predictions(txt) https://drive.google.com/drive/folders/1EUHpg5HvmO3_t1_ePnpn_Xh_Qx2esSmI?usp=sharing

### 5.3 Swin-Transformer
model https://drive.google.com/file/d/1F__reJ77RgZG-nNBTLo4A-BFB0iWL6g2/view?usp=drive_link
predictions(json) https://drive.google.com/file/d/1FEW7It-ovNQhuhTNoP6uVGQFAcoBNBMw/view?usp=drive_link
predictions(txt) https://drive.google.com/drive/folders/1oMAeLh33Uq8GhfamzOE74zDuz-A33u5u?usp=drive_link

### 5.4 YOLOv8l
model https://drive.google.com/file/d/17pFkqSNMrkcq6tBterEs3ERYrzcLrw9y/view?usp=drive_link
predictions(json) https://drive.google.com/file/d/1x80v5vnpkI3VZoYOejoBRJtFWzcjqiMC/view?usp=drive_link
predictions(txt) https://drive.google.com/drive/folders/1N8MHdpGDx6G-umlNhidyRhKZylibBD2Z?usp=drive_link

### 5.5 YOLOv9e
model https://drive.google.com/file/d/1NixthCPC-fGP49b1MIqWDIxuCN7aghBa/view?usp=drive_link
predictions(json) https://drive.google.com/file/d/1oRH9q0ChlOn9dDiuw2og915za11Dy4Sd/view?usp=drive_link
predictions(txt) https://drive.google.com/drive/folders/12z-jld4mJqxEUirjZ28EV9KUBnFtV44I?usp=drive_link

### 5.6 DETR
model https://drive.google.com/file/d/1C-FlDye77Ls4QrGM-b5bJpvVJr_VkkV-/view?usp=drive_link
predictions(json) https://drive.google.com/file/d/1y1gQdNi87HGZIBZ2tceY5lgQXd_NV28p/view?usp=drive_link
predictions(txt) https://drive.google.com/drive/folders/11sxx2nmjkHichUHrET8Ul-7Rgr6TyBMy?usp=drive_link

### 5.7 Deformable-DETR
model https://drive.google.com/file/d/1FITK7HNKJmNCk0VBUyGpJW_Wk5j0g95i/view?usp=drive_link
predictions(json) https://drive.google.com/file/d/1zqyPed8Xl5_qmhxODsJWLvNEXBJRDUI9/view?usp=drive_link
predictions(txt) https://drive.google.com/drive/folders/1dVxeSCItIUQverQo3tjY6honfAajHMpJ?usp=drive_link

### 5.8 Conditional-DETR
model https://drive.google.com/file/d/1nlxDYCYTYtickpgzR25oQ_i381OFQcey/view?usp=drive_link
predictions(json) https://drive.google.com/file/d/1KxM3_crED8_FsCtBfVIU0GTnnv6fXA5s/view?usp=drive_link
predictions(txt) https://drive.google.com/drive/folders/136jd7iVMfINEjW_adVvyMNUdYKLRsWi0?usp=drive_link

### 5.9 DAB-DETR
model https://drive.google.com/file/d/12OFMCoaiJ1HO3lM1faFL4LdU15DShjO_/view?usp=drive_link
predictions(json) https://drive.google.com/file/d/1pYBHuoPZz-_mj5hUIO4bd59IqvenIo5q/view?usp=drive_link
predictions(txt) https://drive.google.com/drive/folders/1npthg61vjSyG8DBOheBznH-PvvdCXxgO?usp=drive_link

### 5.10 DINO-DETR
model https://drive.google.com/file/d/14WSFPHTsApAhq5ZOf7zTEdWinRwlz67v/view?usp=drive_link
predictions(json) https://drive.google.com/file/d/1CyyicIdlm33TR86tVvJZP-gfLGRU5s6C/view?usp=drive_link
predictions(txt) https://drive.google.com/drive/folders/1peDKsPFuZXqmykfGog0kcmyUGR0W-T1v?usp=drive_link

### 5.11 RT-DETR
model https://drive.google.com/file/d/1edjoLoPpZasucKphOtR2lD4MWS0BX5Ok/view?usp=drive_link
predictions(json) https://drive.google.com/file/d/1MnaKxrU3-eLxeEl4yYsl-R2mh03iEfT8/view?usp=drive_link
predictions(txt) https://drive.google.com/drive/folders/1u_2VPbLf-CIbPxYQB_RZNAwTz6YIFrck?usp=drive_link

### 5.12 F2DNet
model https://drive.google.com/file/d/1MqvUcsq6GiUkjRx43tO2KX7JfETKAhqQ/view?usp=drive_link
predictions(json) https://drive.google.com/file/d/1HiZgdMvsxAXCokWcJyxuOxkw1tmC9zR0/view?usp=drive_link
predictions(txt) https://drive.google.com/drive/folders/1mJA4L5nKVySZmgpV5NjG61DX0udenueE?usp=drive_link
