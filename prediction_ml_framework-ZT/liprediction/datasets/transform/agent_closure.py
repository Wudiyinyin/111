# Copyright (c) 2021 Li Auto Company. All rights reserved.

import torch


def size_repr(key, item, indent=0):
    indent_str = ' ' * indent
    if torch.is_tensor(item) and item.dim() == 0:
        out = f'{item.item()}, {item.dtype}, {item.device}'
    elif torch.is_tensor(item):
        out = f'{str(list(item.size()))}, {item.dtype}, {item.device}'
    elif isinstance(item, list) or isinstance(item, tuple):
        out = str([len(item)])
    elif isinstance(item, dict):
        lines = [indent_str + size_repr(k, v, 2) for k, v in item.items()]
        out = '{\n' + ',\n'.join(lines) + '\n' + indent_str + '}'
    elif isinstance(item, str):
        out = f'"{item}"'
    else:
        out = str(item)

    return f'{indent_str}{key}={out}'


class AgentClosure(object):

    def __init__(self, **kwarg):
        for key, value in kwarg.items():
            setattr(self, key, value)

    @property
    def keys(self):
        return [
            key for key in self.__dict__.keys()
            if self.__dict__[key] is not None and key[:2] != '__' and key[-2:] != '__'
        ]

    def apply(self, func):
        """
        apply func to all the tensor element
        """

        for key in self.keys:
            self[key] = self.__apply__(self[key], func)

        return self

    def to(self, device, **kwargs):
        return self.apply(lambda x: x.to(device, **kwargs))

    def contiguous(self):
        return self.apply(lambda x: x.contiguous())

    def __apply__(self, item, func):
        """
        apply func to tensor, if item is list, tuple or dict, apply it recursively
        """

        if torch.is_tensor(item):
            return func(item)
        elif isinstance(item, (tuple, list)):
            return [self.__apply__(v, func) for v in item]
        elif isinstance(item, dict):
            return {k: self.__apply__(v, func) for k, v in item.items()}
        else:
            return item

    def __getitem__(self, key):
        return getattr(self, key, None)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __iter__(self):
        for key in sorted(self.keys):
            yield key, self[key]

    def __repr__(self):
        cls = str(self.__class__.__name__)
        has_dict = any([isinstance(item, dict) for _, item in self])

        if not has_dict:
            info = [size_repr(key, item) for key, item in self]
            return '{}({})'.format(cls, '; '.join(info))
        else:
            info = [size_repr(key, item, indent=2) for key, item in self]
            return '{}(\n{}\n)'.format(cls, ';\n'.join(info))


class AgentClosureBatch(AgentClosure):

    def __init__(self, **kwargs):
        self.size = 0
        super().__init__(**kwargs)

    def __len__(self):
        return self.size

    def to_data_list(self):
        # exclude batch specific keys
        keys = [key for key in self.keys if key not in ['slice_size', 'cumsum', 'size']]

        # split batch into chuncks
        batch = {}
        for key in keys:
            value = self[key]
            slice_size = self.slice_size[key]
            cat_dim = self.__cat_dim__(key, value)
            if isinstance(value, torch.Tensor):
                batch[key] = torch.split(value, slice_size, dim=cat_dim)
            elif isinstance(value, list) and isinstance(value[0], torch.Tensor):
                array_list = [torch.split(value[i], slice_size[i], cat_dim[i]) for i in range(len(value))]
                batch[key] = [[array[i] for array in array_list] for i in range(len(self.size))]
            else:
                batch[key] = value

        # construct data list
        data_list = []
        for i in range(len(self)):
            data = AgentClosure()
            for key in keys:
                item = batch[key][i]
                cumsum = self.cumsum[key][i]
                if isinstance(item, torch.Tensor) and item.dtype != torch.bool:
                    item = item - cumsum
                elif isinstance(item, list) and isinstance(item[0], torch.Tensor) and item[0].dtype != torch.bool:
                    item = [item[k] - cumsum[k] for k in range(len(item))]
                data[key] = item
            data_list.append(data)

        return data_list

    @classmethod
    def from_data_list(cls, data_list):
        """
        Make a data list into a batch
        """

        # use data_list[0] to get meta info
        ref_data = data_list[0]

        # get all the attributes name we care
        keys = ref_data.keys

        # store the value of same key in a list
        batch = {k: [] for k in keys}
        cumsum = {key: [cls.__init_cumsum__(key, ref_data)] for key in keys}
        slice_size = {key: [] for key in keys}
        for i, data in enumerate(data_list):
            for key in keys:
                item = data[key]
                cum = cumsum[key][-1]

                # treat 0-dimensional tensors as 1-dimensional
                if isinstance(item, torch.Tensor) and item.dim() == 0:
                    item = item.unsqueeze(0)

                # if is a tensor
                if isinstance(item, torch.Tensor) and item.dtype != torch.bool:
                    item = item + cum

                # if is a list of tensor
                if isinstance(item, list) and isinstance(item[0], torch.Tensor) and item[0].dtype != torch.bool:
                    new_item = []
                    for element_id, element in enumerate(item):
                        if element.dim() == 0:
                            element = element.unsqueeze(0)
                        element = element + cum[element_id]
                        new_item.append(element)
                    item = new_item

                # store the processed item
                batch[key].append(item)

                # update cumsum
                inc = cls.__inc__(key, data)
                if isinstance(cum, list):
                    new_cum = [cum[i] + inc[i] for i in range(len(cum))]
                else:
                    new_cum = cum + inc
                cumsum[key].append(new_cum)

                # update slice_size
                cat_dim = cls.__cat_dim__(key, data)
                if isinstance(item, torch.Tensor):
                    slice_size[key].append(item.shape[cat_dim])
                elif isinstance(item, list) and isinstance(item[0], torch.Tensor):
                    slice_size[key].append([item[i].shape[cat_dim[i]] for i in range(len(item))])
                else:
                    slice_size[key].append(1)

        # construct final batch
        for key, item_list in batch.items():
            cat_dim = cls.__cat_dim__(key, ref_data)
            if isinstance(item_list[0], torch.Tensor):
                batch[key] = torch.cat(item_list, cat_dim)
            elif isinstance(item_list[0], list) and isinstance(item_list[0][0], torch.Tensor):
                # item_list = [[v0,v1,...], [v0,v1,...]]
                batch[key] = [torch.cat([item[i] for item in item_list], cat_dim[i]) for i in range(len(item_list[0]))]

        batch['cumsum'] = cumsum
        batch['slice_size'] = slice_size
        batch['size'] = len(data_list)

        return cls(**batch)

    @classmethod
    def __inc__(cls, key, data=None):
        """
        inherited class would override this
        """
        return 0

    @classmethod
    def __cat_dim__(cls, key, data=None):
        """
        inherited class would override this
        """
        return 0

    @classmethod
    def __init_cumsum__(cls, key, data=None):
        """
        inherited class would override this
        """
        return 0
