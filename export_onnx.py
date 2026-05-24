import torch
import os

# 1. 直接导入仓库里的 AnimeGAN 插件模型
from anime_model import Generator as AnimeGenerator
# 2. 直接导入仓库里的 CycleGAN 官方网络模块
from models.networks import define_G


def convert_pt_to_onnx(model, pt_path, onnx_name, img_size):
    device = torch.device("cpu")
    print(f"正在读取: {pt_path}")

    state_dict = torch.load(pt_path, map_location=device)
    if 'model' in state_dict:
        state_dict = state_dict['model']

    model.load_state_dict(state_dict)
    model.eval()

    dummy_input = torch.randn(1, 3, img_size, img_size)
    onnx_path = os.path.join("onnx_models", onnx_name)
    os.makedirs("onnx_models", exist_ok=True)

    print("正在转换为 ONNX 静态图...")
    with torch.no_grad():
        traced_model = torch.jit.trace(model, dummy_input)

    torch.onnx.export(
        traced_model, dummy_input, onnx_path,
        export_params=True, opset_version=12, do_constant_folding=True,
        input_names=['input_image'], output_names=['output_image'],
        dynamic_axes={'input_image': {2: 'height', 3: 'width'},
                      'output_image': {2: 'height', 3: 'width'}},
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
        dynamo=False
    )
    print(f"转换成功！生成了: {onnx_path}\n")


if __name__ == "__main__":
    # 导出 AnimeGAN
    # anime_net = AnimeGenerator()
    # convert_pt_to_onnx(anime_net, "anime_weights/paprika.pt", "anime_style.onnx", 512)
    #
    # # 导出你的 CycleGAN (使用 CycleGAN 原生函数生成网络骨架)
    # try:
    #     my_cycle_net = define_G(3, 3, 64, 'resnet_9blocks', norm='instance', use_dropout=False, init_type='normal',
    #                             init_gain=0.02, gpu_ids=[])
    #     convert_pt_to_onnx(my_cycle_net, "checkpoints/apple2orange_cyclegan/latest_net_G_A.pth", "my_cyclegan.onnx",
    #                        256)
    # except Exception as e:
    #     print("CycleGAN 尚未训练完成，仅导出了 Anime 模型。")
    if __name__ == "__main__":
        # ==========================================
        # 1. 导出 AnimeGAN (街景 -> 动漫)
        # ==========================================
        try:
            anime_net = AnimeGenerator()
            convert_pt_to_onnx(anime_net, "anime_weights/paprika.pt", "anime_style.onnx", 512)
        except Exception as e:
            print(f"AnimeGAN 导出失败: {e}")

        # ==========================================
        # 2. 导出 苹果与橘子 (apple2orange)
        # A=苹果, B=橘子。 G_A负责A->B, G_B负责B->A
        # ==========================================
        try:
            net = define_G(3, 3, 64, 'resnet_9blocks', norm='instance', use_dropout=False, init_type='normal',
                           init_gain=0.02)
            convert_pt_to_onnx(net, "checkpoints/apple2orange_cyclegan/latest_net_G_A.pth", "apple_to_orange.onnx", 256)
        except Exception as e:
            print(f"苹果变橘子 导出失败: {e}")

        try:
            net = define_G(3, 3, 64, 'resnet_9blocks', norm='instance', use_dropout=False, init_type='normal',
                           init_gain=0.02)
            convert_pt_to_onnx(net, "checkpoints/apple2orange_cyclegan/latest_net_G_B.pth", "orange_to_apple.onnx", 256)
        except Exception as e:
            print(f"橘子变苹果 导出失败: {e}")

        # ==========================================
        # 3. 导出 莫奈与照片 (monet2photo)
        # A=莫奈画作, B=真实照片。 G_A负责A->B, G_B负责B->A
        # ==========================================
        try:  # 莫奈 -> 照片 (用 G_A)
            net = define_G(3, 3, 64, 'resnet_9blocks', norm='instance', use_dropout=False, init_type='normal',
                           init_gain=0.02)
            convert_pt_to_onnx(net, "checkpoints/monet2photo_cyclegan/latest_net_G_A.pth", "monet_to_photo.onnx", 256)
        except Exception as e:
            print(f"莫奈变照片 导出失败: {e}")

        try:  # 照片 -> 莫奈 (用 G_B)
            net = define_G(3, 3, 64, 'resnet_9blocks', norm='instance', use_dropout=False, init_type='normal',
                           init_gain=0.02)
            convert_pt_to_onnx(net, "checkpoints/monet2photo_cyclegan/latest_net_G_B.pth", "photo_to_monet.onnx", 256)
        except Exception as e:
            print(f"照片变莫奈 导出失败: {e}")

        # ==========================================
        # 4. 导出 手机与单反 (iphone2dslr)
        # A=手机照片, B=单反照片。 G_A负责A->B, G_B负责B->A
        # ==========================================
        try:  # 手机 -> 单反 (用 G_A)
            net = define_G(3, 3, 64, 'resnet_9blocks', norm='instance', use_dropout=False, init_type='normal',
                           init_gain=0.02)
            convert_pt_to_onnx(net, "checkpoints/iphone2dslr_cyclegan/latest_net_G_A.pth", "iphone_to_dslr.onnx", 256)
        except Exception as e:
            print(f"手机变单反 导出失败: {e}")

        try:  # 单反 -> 手机 (用 G_B)
            net = define_G(3, 3, 64, 'resnet_9blocks', norm='instance', use_dropout=False, init_type='normal',
                           init_gain=0.02)
            convert_pt_to_onnx(net, "checkpoints/iphone2dslr_cyclegan/latest_net_G_B.pth", "dslr_to_iphone.onnx", 256)
        except Exception as e:
            print(f"单反变手机 导出失败: {e}")