# load and extract all faces
directory = 'img_align_celeba/img_align_celeba/'
all_faces = load_faces(directory, 10000)
print('Loaded: ', all_faces.shape)

# save in compressed format
savez_compressed('img_align_celeba_128.npz', all_faces)
