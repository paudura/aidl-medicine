def show_dicom_images_with_boxes(dd):
    f, ax = plt.subplots(2,2, figsize=(16,18))
    for i  in range(dd.shape[0]):
        patientImage = dd.loc[i, 'patientId']+'.dcm'
        imagePath = os.path.join("/home/medicine_project/input_data/stage_2_train_images/",patientImage)
        data_row_img_data = dcm.read_file(imagePath)
        modality = data_row_img_data.Modality
        age = data_row_img_data.PatientAge
        sex = data_row_img_data.PatientSex
        data_row_img = dcm.dcmread(imagePath)
        ax[i//3, i%3].imshow(data_row_img.pixel_array, cmap=plt.cm.bone) 
        ax[i//3, i%3].axis('off')
        ax[i//3, i%3].set_title('ID: {}\nModality: {} Age: {} Sex: {} Target: {}\nClass: {}'.format(
                dd.loc[i, 'patientId'],modality, age, sex, dd.loc[i, 'Target'], dd.loc[i, 'class']))
        

        ax[i//3, i%3].add_patch(Rectangle(xy=(dd.loc[i, "x"], dd.loc[i, "y"]),
                    width=dd.loc[i, "width"],height=dd.loc[i, "height"], 
                    color="yellow",alpha = 0.1))  

        circle1 = plt.Circle((dd.loc[i, "x"], dd.loc[i, "y"]), 0.5, color='r')

        ax[i//3, i%3].add_patch(circle1)
        plt.show()
        plt.savefig('books_read.png')
