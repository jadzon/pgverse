import os
import sys
from PIL import Image
from PIL import ImageChops
from tqdm import tqdm
def create_dataset_from_template(ref_image, output_path):
    os.mkdir(output_path)
    i = 0
    total_iterations = len(range(-150, 150, 30)) * len(range(-150, 150, 30)) * len(range(0, 180, 2))
    with tqdm(total=total_iterations, desc="Creating dataset in " + output_path) as pbar:
        for offset_value_x in range(-150, 150, 30):
            for offset_value_y in range(-150, 150, 30):
                offset_img = ImageChops.offset(ref_image, offset_value_x, offset_value_y)
                for rot_value in range(0, 180, 2):
                    new_img = offset_img.rotate(-rot_value)
                    new_img.save(output_path + "/"+ str(i) + ".png", "PNG")
                    i += 1
                    pbar.update(1)



def main():
    dataset_path = "schematic_symbols/dataset"
    ref_image_path = "schematic_symbols/template"
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)

    resistor_path = os.path.join(dataset_path, "resistor")
    capacitor_path = os.path.join(dataset_path, "capacitor")
    inductor_path = os.path.join(dataset_path, "inductor")
    voltage_source_path = os.path.join(dataset_path, "voltage_source")
    current_source_path = os.path.join(dataset_path, "current_source")
    ground_path = os.path.join(dataset_path, "ground")
    #transistor_path = os.path.join(dataset_path, "transistor")
    amplifier_path = os.path.join(dataset_path, "amplifier")
    diode_path = os.path.join(dataset_path, "diode")
    
    template_resistor = Image.open(ref_image_path + "/resistor.png")

    template_capacitor = Image.open(ref_image_path + "/capacitor.png")
    template_inductor = Image.open(ref_image_path + "/inductor.png")
    template_voltage_source = Image.open(ref_image_path + "/voltage_source.png")
    template_current_source = Image.open(ref_image_path + "/current_source.png")
    template_ground = Image.open(ref_image_path + "/ground.png")
    #template_transistor = Image.open(ref_image_path + "/transistor.png")
    #template_amplifier = Image.open(ref_image_path + "/amplifier.png")
    template_diode = Image.open(ref_image_path + "/diode.png")

    create_dataset_from_template(template_resistor, resistor_path)
    create_dataset_from_template(template_capacitor, capacitor_path)
    create_dataset_from_template(template_inductor, inductor_path)
    create_dataset_from_template(template_voltage_source, voltage_source_path)
    create_dataset_from_template(template_current_source, current_source_path)
    create_dataset_from_template(template_ground, ground_path)
    #create_dataset_from_template(template_transistor, transistor_path)
    #create_dataset_from_template(template_amplifier, amplifier_path)
    create_dataset_from_template(template_diode, diode_path)


if __name__ == '__main__':
    main()