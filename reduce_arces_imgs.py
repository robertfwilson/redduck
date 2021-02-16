from sea_sea_dee import correct_all_imgs


def run_arces_corrections(data_dir, fnames, bias_fname):

	correct_all_imgs(data_dir, img_fname, bias_fname, dark_fname=None, flat_fname=None, dark_exptime=1e6, sci_exptime=1e6, texp_key='EXPTIME', gain=3.8,pixel_mask=None, fringe_frame=None, clean_cosmicrays=False,)

	print('ALL DONE')


if __name__=="__main__":

	data_dir=''
	img_fname=''
	bias_fname=''

	run_arces_corrections(data_dir,img_fname,bias_fname)


