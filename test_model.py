import torch.optim
from Load_Dataset import ValGenerator, ImageToImage2D
from torch.utils.data import DataLoader
import warnings
import pandas as pd
warnings.filterwarnings("ignore")
import Config as config
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from nets.LViT import LViT
from utils import *
import cv2
from skimage.measure import label, regionprops

def jaccard_score(y_true, y_pred):
    """
    计算 Jaccard Index (IoU)
    :param y_true: 真实掩码（展平后的 1D 数组）
    :param y_pred: 预测掩码（展平后的 1D 数组）
    :return: IoU 值
    """
    intersection = np.sum(y_true * y_pred)          # 交集
    union = np.sum(y_true) + np.sum(y_pred) - intersection  # 并集
    return intersection / (union + 1e-7)             # 避免除零
#测试测试测试
def calculate_lesion_stats(predict_mask):
    """
    计算病灶统计信息
    :param predict_mask: 预测的二值掩码(0-1)
    :return: 病灶数量、总面积、最大病灶面积、平均面积
    """
    labeled_mask = label(predict_mask)
    regions = regionprops(labeled_mask)

    num_lesions = len(regions)
    total_area = sum(region.area for region in regions)
    max_area = max(region.area for region in regions) if regions else 0
    avg_area = total_area / num_lesions if num_lesions > 0 else 0

    return num_lesions, total_area, max_area, avg_area


def show_image_with_dice(predict_save, labs, save_path):
    tmp_lbl = (labs).astype(np.float32)
    tmp_3dunet = (predict_save).astype(np.float32)
    dice_pred = 2 * np.sum(tmp_lbl * tmp_3dunet) / (np.sum(tmp_lbl) + np.sum(tmp_3dunet) + 1e-5)
    iou_pred = jaccard_score(tmp_lbl.reshape(-1), tmp_3dunet.reshape(-1))

    # 计算病灶统计信息
    num_lesions, total_area, max_area, avg_area = calculate_lesion_stats(predict_save)

    if config.task_name == "MoNuSeg":
        predict_save = cv2.pyrUp(predict_save, (448, 448))
        predict_save = cv2.resize(predict_save, (2000, 2000))
        cv2.imwrite(save_path, predict_save * 255)
    else:
        cv2.imwrite(save_path, predict_save * 255)

    return dice_pred, iou_pred, num_lesions, total_area, max_area, avg_area


def vis_and_save_heatmap(model, input_img, text, img_RGB, labs, vis_save_path, dice_pred, dice_ens):
    model.eval()
    output = model(input_img.cuda(), text.cuda())
    pred_class = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output))
    predict_save = pred_class[0].cpu().data.numpy()
    predict_save = np.reshape(predict_save, (config.img_size, config.img_size))

    # 获取新增的病灶统计信息
    dice_pred_tmp, iou_tmp, num_lesions, total_area, max_area, avg_area = show_image_with_dice(
        predict_save, labs, save_path=vis_save_path + '_predict' + model_type + '.jpg')

    return dice_pred_tmp, iou_tmp, num_lesions, total_area, max_area, avg_area


# 新增函数：调用LLM生成报告
def generate_medical_report(num_lesions, total_area, max_area, avg_area):
    """
    调用LLM生成病历风格报告
    :param num_lesions: 病灶数量
    :param total_area: 总面积(像素)
    :param max_area: 最大病灶面积(像素)
    :param avg_area: 平均病灶面积(像素)
    :return: 自然语言报告文本
    """
    # 实际应用中这里应替换为真实LLM API调用
    # 示例使用模拟函数，实际实现需接入真实模型

    # 提示词模板（示例）
    prompt = f"""
请根据以下肺部CT病灶分析结果，生成一份专业的放射科诊断报告：
- 病灶数量：{num_lesions}
- 病灶总面积：{total_area} 平方像素
- 最大病灶面积：{max_area} 平方像素
- 平均病灶面积：{avg_area:.1f} 平方像素

报告要求：
1. 采用标准放射科报告格式（描述、印象）
2. 语言专业简洁，使用医学术语
3. 对临床意义进行适当解释
4. 如发现异常需给出处理建议
"""
    # 实际LLM调用代码
    # response = llm.generate(prompt)
    # report = response['choices'][0]['text']

    # 模拟响应（实际使用应注释掉）
    report = f"""
**CT肺部平扫影像分析报告**

**检查描述：**
双肺野可见{num_lesions}个局灶性异常密度影，病灶分布以胸膜下区为主。最大病灶位于右下肺后基底段，截面大小约{max_area / 100:.1f}mm×{max_area / 100 * 0.8:.1f}mm（按像素比例估算）。病灶形态呈结节状，边缘尚清，平均密度值约-450HU。邻近胸膜轻度牵拉，病灶间无融合征象。

**印象：**
1. 双肺多发小结节，考虑为炎性肉芽肿可能，建议随访复查（3个月）
2. 右下肺较大结节需密切观察，建议结合增强CT进一步评估
"""
    return report.strip()
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    test_session = config.test_session

    if config.task_name == "MoNuSeg":
        test_num = 14
        model_type = config.model_name
        model_path = "./MoNuSeg/" + model_type + "/" + test_session + "/models/best_model-" + model_type + ".pth.tar"

    elif config.task_name == "MosMedDataPlus":
        test_num = 273
        model_type = config.model_name
        model_path = "./MosMedDataPlus/" + model_type + "/" + test_session + "/models/best_model-" + model_type + ".pth.tar"

    save_path = config.task_name + '/' + model_type + '/' + test_session + '/'
    vis_path = "./" + config.task_name + '_visualize_test/'
    if not os.path.exists(vis_path):
        os.makedirs(vis_path)

        # 创建结果DataFrame，增加病灶统计列
    results_df = pd.DataFrame(columns=[
        'Image',
        'Description',
        'Dice',
        'IoU',
        'Num_Lesions',
        'Total_Area(pixels)',
        'Max_Lesion_Area',
        'Avg_Lesion_Area'
    ])
    checkpoint = torch.load(model_path, map_location='cuda')

    if model_type == 'LViT':
        config_vit = config.get_CTranS_config()
        model = LViT(config_vit, n_channels=config.n_channels, n_classes=config.n_labels)

    elif model_type == 'LViT_pretrain':
        config_vit = config.get_CTranS_config()
        model = LViT(config_vit, n_channels=config.n_channels, n_classes=config.n_labels)


    else:
        raise TypeError('Please enter a valid name for the model type')

    model = model.cuda()
    if torch.cuda.device_count() > 1:
       print("Let's use {0} GPUs!".format(torch.cuda.device_count()))
       model = nn.DataParallel(model)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    print('Model loaded !')
    tf_test = ValGenerator(output_size=[config.img_size, config.img_size])
    test_text = read_text(config.test_dataset + 'Test_text.xlsx')
    test_dataset = ImageToImage2D(config.test_dataset, config.task_name, test_text, tf_test, image_size=config.img_size)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    dice_pred = 0.0
    iou_pred = 0.0
    dice_ens = 0.0

    with tqdm(total=test_num, desc='Test visualize', unit='img', ncols=70, leave=True) as pbar:
        for i, (sampled_batch, names) in enumerate(test_loader, 1):
            test_data, test_label, test_text = sampled_batch['image'], sampled_batch['label'], sampled_batch['text']
            arr = test_data.numpy()
            arr = arr.astype(np.float32())
            lab = test_label.data.numpy()
            img_lab = np.reshape(lab, (lab.shape[1], lab.shape[2])) * 255
            fig, ax = plt.subplots()
            plt.imshow(img_lab, cmap='gray')
            plt.axis("off")
            height, width = config.img_size, config.img_size
            fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.savefig(vis_path + str(names) + "_lab.jpg", dpi=300)
            plt.close()
            input_img = torch.from_numpy(arr)
            dice_pred_t, iou_pred_t, num_lesions, total_area, max_area, avg_area = vis_and_save_heatmap(
                model, input_img, test_text, None, lab,
                vis_path + str(names),
                dice_pred=dice_pred, dice_ens=dice_ens)

            # 添加结果到DataFrame
            new_row = {
                'Image': str(names[0]),
                'Description': generate_medical_report(num_lesions, total_area, max_area, avg_area),
                'Dice': dice_pred_t,
                'IoU': iou_pred_t,
                'Num_Lesions': num_lesions,
                'Total_Area(pixels)': total_area,
                'Max_Lesion_Area': max_area,
                'Avg_Lesion_Area': avg_area
            }
            results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)

            dice_pred += dice_pred_t
            iou_pred += iou_pred_t
            torch.cuda.empty_cache()
            pbar.update()

    # 计算并添加平均指标
    avg_row = {
        'Image': 'Average',
        'Description': '-',
        'Dice': dice_pred / test_num,
        'IoU': iou_pred / test_num,
        'Num_Lesions': results_df['Num_Lesions'].mean(),
        'Total_Area(pixels)': results_df['Total_Area(pixels)'].mean(),
        'Max_Lesion_Area': results_df['Max_Lesion_Area'].mean(),
        'Avg_Lesion_Area': results_df['Avg_Lesion_Area'].mean()
    }
    results_df = pd.concat([results_df, pd.DataFrame([avg_row])], ignore_index=True)

    # 保存结果到Excel文件
    results_path = os.path.join(vis_path, 'prediction_results.xlsx')
    results_df.to_excel(results_path, index=False)

    print("dice_pred", dice_pred / test_num)
    print("iou_pred", iou_pred / test_num)
    print(f"预测结果已保存到: {results_path}")
