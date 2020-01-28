from supereeg import Brain
import copy

class BandBrain(Brain):
    def __init__(self, bo, og_bo):
        super().__init__(bo)
        self.og_bo = og_bo

    # def apply_filter(self, inplace=True):
    #     """ Return a filtered copy """

    #     if self.filter is None:
    #         if not inplace:
    #             return copy.deepcopy(self)
    #         else:
    #             return None

    #     x = copy.copy(self.__dict__)
    #     x['data'] = self.get_data()
    #     x['locs'] = self.get_locs()
    #     x['kurtosis'] = self.og_bo.kurtosis

    #     if self.filter == 'kurtosis':
    #         x['kurtosis'] = x['kurtosis'][x['kurtosis'] <= x['kurtosis_threshold']]

    #     for key in ['n_subs', 'n_elecs', 'n_sessions', 'dur', 'filter_inds', 'nifti_shape']:
    #         if key in x.keys():
    #             x.pop(key)

    #     boc = Brain(**x)
    #     boc = BandBrain(boc, self.og_bo)
    #     boc.filter = None
    #     boc.update_info()
    #     if inplace:
    #         self = boc
    #     else:
    #         return boc

    def get_data(self):
        """
        Gets data from brain object
        """
        self.og_bo.update_filter_inds()
        return self.data.iloc[:, self.og_bo.filter_inds.ravel()].reset_index(drop=True)

    def get_locs(self, og_bo=None):
        """
        Gets locations from brain object
        """
        self.og_bo.update_filter_inds()
        return self.locs.iloc[self.og_bo.filter_inds.ravel(), :].reset_index(drop=True)


# from supereeg import Brain
# import copy

# class BandBrain(Brain):
#     def __init__(self, bo, og_bo=None, og_filter_inds=None):
#         super().__init__(bo)
#         self.og_filter_inds = og_filter_inds
#         self.og_bo = og_bo

#     """ A brain sublass that filters electrodes based on the original .bo (og_bo) """
#     def apply_filter(self, og_bo=None, inplace=True):
#         """ Return a filtered copy """
#         if og_bo is None:
#             og_bo = self.og_bo

#         if self.filter is None:
#             if not inplace:
#                 return copy.deepcopy(self)
#             else:
#                 return None

#         x = copy.copy(self.__dict__)
#         x['data'] = self.get_data()
#         x['locs'] = self.get_locs()
#         x['kurtosis'] = og_bo.kurtosis

#         if self.filter == 'kurtosis':
#             x['kurtosis'] = x['kurtosis'][x['kurtosis'] <= x['kurtosis_threshold']]

#         for key in ['n_subs', 'n_elecs', 'n_sessions', 'dur', 'filter_inds', 'nifti_shape', 'og_bo']:
#             if key in x.keys():
#                 x.pop(key)

#         boc = Brain(**x)
#         boc.filter = None
#         boc.update_info()
#         if inplace:
#             self.__init__(boc, og_bo)
#         else:
#             return boc

#     def get_data(self, og_bo=None):
#         """
#         Gets data from brain object
#         """
#         if og_bo is not None:
#             og_bo.update_filter_inds()
#         try:
#             filter_inds = self.og_bo.filter_inds
#         except:
#             filter_inds = self.og_filter_inds
#         return self.data.iloc[:, self.og_filter_inds.ravel()].reset_index(drop=True)

#     def get_locs(self, og_bo=None):
#         """
#         Gets locations from brain object
#         """
#         if og_bo is not None:
#             og_bo.update_filter_inds()
#         try:
#             filter_inds = self.og_bo.filter_inds
#         except:
#             filter_inds = self.og_filter_inds
#         return self.locs.iloc[self.og_filter_inds.ravel(), :].reset_index(drop=True)
