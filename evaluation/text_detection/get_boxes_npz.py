import os, argparse
import glob
import subprocess


def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"文件夹 {folder_path} 已创建")
    else:
        print(f"文件夹 {folder_path} 已存在")


def call_dbnet_infer(
    exec_file,
    model_prefix,
    vdsp_params,
    device_id,
    image_path,
    out_npz_dir,
    elf_file,
    use_custom_op,
):
    cmd = [
        exec_file,
        "--model_prefix",
        model_prefix,
        "--vdsp_params",
        vdsp_params,
        "--display_boxes",
        "1",
        "--input_file",
        image_path,
        "--device_id",
        device_id,
        "--output_npz",
        os.path.join(out_npz_dir, os.path.basename(img_path).split(".")[0] + ".npz"),
        "--elf_file",
        elf_file,
    ]

    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
    )

    output, error = process.communicate()
    print(output)

    # 检查命令是否执行成功
    if process.returncode == 0:
        print(cmd)
        print("命令执行成功")
    else:
        print("命令执行失败")
        print(cmd)
        exit()
        # os.system(cmd)


if __name__ == "__main__":

    parse = argparse.ArgumentParser(description="EVAL")
    parse.add_argument(
        "--exec_file",
        type=str,
        default="/home/zhchen/vastpipe-samples/build/vastpipe-samples/bin/dbnet_sample",
    )
    parse.add_argument(
        "--model_prefix",
        type=str,
        default="/home/zhchen/vastpipe-samples/data/dbnet_mobilenetv3-int8-kl_divergence-1_3_736_1280-debug/dbnet_mobilenetv3",
    )
    parse.add_argument(
        "--vdsp_params",
        type=str,
        default="/home/zhchen/vastpipe-samples/data/dbnet_resnet50_vd-int8-kl_divergence-1_3_736_1280-vacc/dbnet_resnet50_vd_rgb888_vdsp.json",
    )
    parse.add_argument("--device_id", type=str, default="0")
    parse.add_argument(
        "--input_file_path",
        type=str,
        default="/home/zhchen/vastpipe-samples/data/ch4_test_images",
    )
    parse.add_argument("--out_npz_dir", type=str, default="boxes_npz")
    parse.add_argument("--use_custom_op", type=str, default="0")
    parse.add_argument("--elf_file", type=str, default="")
    args = parse.parse_args()

    create_folder_if_not_exists(args.out_npz_dir)

    imgs_path = args.input_file_path

    # 使用通配符 * 来匹配文件夹下的所有文件
    file_list = glob.glob(imgs_path + "/*.jpg")

    # 输出文件列表
    # exec_file, model_prefix, vdsp_params , device_id , image_path, out_npz_dir
    for img_path in file_list:
        call_dbnet_infer(
            args.exec_file,
            args.model_prefix,
            args.vdsp_params,
            args.device_id,
            img_path,
            args.out_npz_dir,
            args.elf_file,
            args.use_custom_op,
        )
