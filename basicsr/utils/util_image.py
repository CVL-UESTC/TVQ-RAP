import torch

class ImageSpliterTh:
    def __init__(self, im, pch_size, stride, sf=1, extra_bs=1):
        '''
        Input:
            im: n x c x h x w, torch tensor, float, low-resolution image in SR
            pch_size, stride: patch setting
            sf: scale factor in image super-resolution
            pch_bs: aggregate pchs to processing, only used when inputing single image
        '''
        assert stride <= pch_size
        self.stride = stride
        self.pch_size = pch_size
        self.sf = sf
        self.extra_bs = extra_bs

        bs, chn, height, width= im.shape
        self.true_bs = bs

        self.height_starts_list = self.extract_starts(height)
        self.width_starts_list = self.extract_starts(width)
        self.starts_list = []
        for ii in self.height_starts_list:
            for jj in self.width_starts_list:
                self.starts_list.append([ii, jj])

        self.length = self.__len__()
        self.count_pchs = 0

        self.im_ori = im
        self.im_res = torch.zeros([bs, chn, height*sf, width*sf], dtype=im.dtype, device=im.device)
        self.pixel_count = torch.zeros([bs, chn, height*sf, width*sf], dtype=im.dtype, device=im.device)

    def extract_starts(self, length):
        if length <= self.pch_size:
            starts = [0,]
        else:
            starts = list(range(0, length, self.stride))
            for ii in range(len(starts)):
                if starts[ii] + self.pch_size > length:
                    starts[ii] = length - self.pch_size
            starts = sorted(set(starts), key=starts.index)
        return starts

    def __len__(self):
        return len(self.height_starts_list) * len(self.width_starts_list)

    def __iter__(self):
        return self

    def __next__(self):
        if self.count_pchs < self.length:
            index_infos = []
            current_starts_list = self.starts_list[self.count_pchs:self.count_pchs+self.extra_bs]
            for ii, (h_start, w_start) in enumerate(current_starts_list):
                w_end = w_start + self.pch_size
                h_end = h_start + self.pch_size
                current_pch = self.im_ori[:, :, h_start:h_end, w_start:w_end]
                if ii == 0:
                    pch =  current_pch
                else:
                    pch = torch.cat([pch, current_pch], dim=0)

                h_start *= self.sf
                h_end *= self.sf
                w_start *= self.sf
                w_end *= self.sf
                index_infos.append([h_start, h_end, w_start, w_end])

            self.count_pchs += len(current_starts_list)
        else:
            raise StopIteration()

        return pch, index_infos

    def update(self, pch_res, index_infos):
        '''
        Input:
            pch_res: (n*extra_bs) x c x pch_size x pch_size, float
            index_infos: [(h_start, h_end, w_start, w_end),]
        '''
        assert pch_res.shape[0] % self.true_bs == 0
        pch_list = torch.split(pch_res, self.true_bs, dim=0)
        assert len(pch_list) == len(index_infos)
        for ii, (h_start, h_end, w_start, w_end) in enumerate(index_infos):
            current_pch = pch_list[ii]
            self.im_res[:, :, h_start:h_end, w_start:w_end] +=  current_pch
            self.pixel_count[:, :, h_start:h_end, w_start:w_end] += 1

    def gather(self):
        assert torch.all(self.pixel_count != 0)
        return self.im_res.div(self.pixel_count)

    # def update(self, pch_res, index_infos):
    #     '''
    #     Input:
    #         pch_res: (n*extra_bs) x c x pch_size x pch_size, float
    #         index_infos: [(h_start, h_end, w_start, w_end),]
    #     '''
    #     assert pch_res.shape[0] % self.true_bs == 0
    #     pch_list = torch.split(pch_res, self.true_bs, dim=0)
    #     assert len(pch_list) == len(index_infos)
    #     for ii, (h_start, h_end, w_start, w_end) in enumerate(index_infos):
    #         current_pch = pch_list[ii]
    #         # print(h_start)
    #         # print(h_end)
    #         self.im_res[:, :, (h_start+224):h_end, (w_start+224):w_end] +=  current_pch[:,:,224:,224:]
    #         # self.pixel_count[:, :, h_start+56:h_end, w_start:w_end] += 1

    # def gather(self):
    #     # assert torch.all(self.pixel_count != 0)
    #     return self.im_res
