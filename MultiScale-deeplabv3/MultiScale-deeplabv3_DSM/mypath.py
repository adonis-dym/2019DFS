class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            return '/path/to/datasets/VOCdevkit/VOC2012/'  # folder that contains VOCdevkit/.
        elif dataset == 'sbd':
            return '/path/to/datasets/benchmark_RELEASE/'  # folder that contains dataset/.
        elif dataset == 'cityscapes':
            return '/path/to/datasets/cityscapes/'     # foler that contains leftImg8bit/
        elif dataset == 'coco':
            return '/path/to/datasets/coco/'
        elif dataset == 'dfc19train':
            return '/data/Public Data/IEEE Data Fusion Contest 2019/DFC2019_track1_trainval/Train2-Track1/'
        elif dataset == 'dfc19val':
            return '/data/Public Data/IEEE Data Fusion Contest 2019/DFC2019_track1_trainval/Split-Val/'
        elif dataset == 'dfc19test':
            return '/data/Public Data/IEEE Data Fusion Contest 2019/DFC2019_track1_trainval/Test-Track1/'#Test-Track1

        
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
