import subprocess
# Function to install packages from inside pytohn
def install(name):
	subprocess.call(['sudo', 'pip3', 'install', name])


install("pandas")

