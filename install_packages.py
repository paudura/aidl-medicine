import subprocess
# Function to install packages from inside pytohn
def install(name):
	subprocess.call(['sudo', 'pip3', 'install', name])


install("pandas")
install('numpy')
install("scikit-image")
install("torchvision")
install("pydicom")
install("sklearn")
install("segmentation-models-pytorch")


#import segmentation_models_pytorch as smp

#model=smp.Unet()
#print(model)