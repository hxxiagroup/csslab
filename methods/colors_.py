'''
颜色支持

- color_scheme 是常用的颜色对应字典
- common_color 是一些常用的离散颜色

使用方法

具体的颜色可以参考下面网站
    https://matplotlib.org/examples/color/colormaps_reference.html
    https://graphistry.github.io/docs/legacy/api/0.9.2/api.html#extendedpalette

datas文件下有color_scheme数据的json版本

'''


ColorScheme = {
'Accent_03' : ['#7fc97f', '#beaed4', '#fdc086'] ,
'Accent_04' : ['#7fc97f', '#beaed4', '#fdc086', '#ffff99'] ,
'Accent_05' : ['#7fc97f', '#beaed4', '#fdc086', '#ffff99', '#386cb0'] ,
'Accent_06' : ['#7fc97f', '#beaed4', '#fdc086', '#ffff99', '#386cb0', '#f0027f'] ,
'Accent_07' : ['#7fc97f', '#beaed4', '#fdc086', '#ffff99', '#386cb0', '#f0027f', '#bf5b17'] ,
'Accent_08' : ['#7fc97f', '#beaed4', '#fdc086', '#ffff99', '#386cb0', '#f0027f', '#bf5b17', '#666666'] ,
'Blues_03' : ['#deebf7', '#9ecae1', '#3182bd'] ,
'Blues_04' : ['#eff3ff', '#bdd7e7', '#6baed6', '#2171b5'] ,
'Blues_05' : ['#eff3ff', '#bdd7e7', '#6baed6', '#3182bd', '#08519c'] ,
'Blues_06' : ['#eff3ff', '#c6dbef', '#9ecae1', '#6baed6', '#3182bd', '#08519c'] ,
'Blues_07' : ['#eff3ff', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#084594'] ,
'Blues_08' : ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#084594'] ,
'Blues_09' : ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b'] ,
'BrBG_03' : ['#d8b365', '#f5f5f5', '#5ab4ac'] ,
'BrBG_04' : ['#a6611a', '#dfc27d', '#80cdc1', '#018571'] ,
'BrBG_05' : ['#a6611a', '#dfc27d', '#f5f5f5', '#80cdc1', '#018571'] ,
'BrBG_06' : ['#8c510a', '#d8b365', '#f6e8c3', '#c7eae5', '#5ab4ac', '#01665e'] ,
'BrBG_07' : ['#8c510a', '#d8b365', '#f6e8c3', '#f5f5f5', '#c7eae5', '#5ab4ac', '#01665e'] ,
'BrBG_08' : ['#8c510a', '#bf812d', '#dfc27d', '#f6e8c3', '#c7eae5', '#80cdc1', '#35978f', '#01665e'] ,
'BrBG_09' : ['#8c510a', '#bf812d', '#dfc27d', '#f6e8c3', '#f5f5f5', '#c7eae5', '#80cdc1', '#35978f', '#01665e'] ,
'BrBG_10' : ['#543005', '#8c510a', '#bf812d', '#dfc27d', '#f6e8c3', '#c7eae5', '#80cdc1', '#35978f', '#01665e', '#003c30'] ,
'BrBG_11' : ['#543005', '#8c510a', '#bf812d', '#dfc27d', '#f6e8c3', '#f5f5f5', '#c7eae5', '#80cdc1', '#35978f', '#01665e', '#003c30'] ,
'BuGn_03' : ['#e5f5f9', '#99d8c9', '#2ca25f'] ,
'BuGn_04' : ['#edf8fb', '#b2e2e2', '#66c2a4', '#238b45'] ,
'BuGn_05' : ['#edf8fb', '#b2e2e2', '#66c2a4', '#2ca25f', '#006d2c'] ,
'BuGn_06' : ['#edf8fb', '#ccece6', '#99d8c9', '#66c2a4', '#2ca25f', '#006d2c'] ,
'BuGn_07' : ['#edf8fb', '#ccece6', '#99d8c9', '#66c2a4', '#41ae76', '#238b45', '#005824'] ,
'BuGn_08' : ['#f7fcfd', '#e5f5f9', '#ccece6', '#99d8c9', '#66c2a4', '#41ae76', '#238b45', '#005824'] ,
'BuGn_09' : ['#f7fcfd', '#e5f5f9', '#ccece6', '#99d8c9', '#66c2a4', '#41ae76', '#238b45', '#006d2c', '#00441b'] ,
'BuPu_03' : ['#e0ecf4', '#9ebcda', '#8856a7'] ,
'BuPu_04' : ['#edf8fb', '#b3cde3', '#8c96c6', '#88419d'] ,
'BuPu_05' : ['#edf8fb', '#b3cde3', '#8c96c6', '#8856a7', '#810f7c'] ,
'BuPu_06' : ['#edf8fb', '#bfd3e6', '#9ebcda', '#8c96c6', '#8856a7', '#810f7c'] ,
'BuPu_07' : ['#edf8fb', '#bfd3e6', '#9ebcda', '#8c96c6', '#8c6bb1', '#88419d', '#6e016b'] ,
'BuPu_08' : ['#f7fcfd', '#e0ecf4', '#bfd3e6', '#9ebcda', '#8c96c6', '#8c6bb1', '#88419d', '#6e016b'] ,
'BuPu_09' : ['#f7fcfd', '#e0ecf4', '#bfd3e6', '#9ebcda', '#8c96c6', '#8c6bb1', '#88419d', '#810f7c', '#4d004b'] ,
'Dark2_03' : ['#1b9e77', '#d95f02', '#7570b3'] ,
'Dark2_04' : ['#1b9e77', '#d95f02', '#7570b3', '#e7298a'] ,
'Dark2_05' : ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e'] ,
'Dark2_06' : ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02'] ,
'Dark2_07' : ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02', '#a6761d'] ,
'Dark2_08' : ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02', '#a6761d', '#666666'] ,
'GnBu_03' : ['#e0f3db', '#a8ddb5', '#43a2ca'] ,
'GnBu_04' : ['#f0f9e8', '#bae4bc', '#7bccc4', '#2b8cbe'] ,
'GnBu_05' : ['#f0f9e8', '#bae4bc', '#7bccc4', '#43a2ca', '#0868ac'] ,
'GnBu_06' : ['#f0f9e8', '#ccebc5', '#a8ddb5', '#7bccc4', '#43a2ca', '#0868ac'] ,
'GnBu_07' : ['#f0f9e8', '#ccebc5', '#a8ddb5', '#7bccc4', '#4eb3d3', '#2b8cbe', '#08589e'] ,
'GnBu_08' : ['#f7fcf0', '#e0f3db', '#ccebc5', '#a8ddb5', '#7bccc4', '#4eb3d3', '#2b8cbe', '#08589e'] ,
'GnBu_09' : ['#f7fcf0', '#e0f3db', '#ccebc5', '#a8ddb5', '#7bccc4', '#4eb3d3', '#2b8cbe', '#0868ac', '#084081'] ,
'Greens_03' : ['#e5f5e0', '#a1d99b', '#31a354'] ,
'Greens_04' : ['#edf8e9', '#bae4b3', '#74c476', '#238b45'] ,
'Greens_05' : ['#edf8e9', '#bae4b3', '#74c476', '#31a354', '#006d2c'] ,
'Greens_06' : ['#edf8e9', '#c7e9c0', '#a1d99b', '#74c476', '#31a354', '#006d2c'] ,
'Greens_07' : ['#edf8e9', '#c7e9c0', '#a1d99b', '#74c476', '#41ab5d', '#238b45', '#005a32'] ,
'Greens_08' : ['#f7fcf5', '#e5f5e0', '#c7e9c0', '#a1d99b', '#74c476', '#41ab5d', '#238b45', '#005a32'] ,
'Greens_09' : ['#f7fcf5', '#e5f5e0', '#c7e9c0', '#a1d99b', '#74c476', '#41ab5d', '#238b45', '#006d2c', '#00441b'] ,
'Greys_03' : ['#f0f0f0', '#bdbdbd', '#636363'] ,
'Greys_04' : ['#f7f7f7', '#cccccc', '#969696', '#525252'] ,
'Greys_05' : ['#f7f7f7', '#cccccc', '#969696', '#636363', '#252525'] ,
'Greys_06' : ['#f7f7f7', '#d9d9d9', '#bdbdbd', '#969696', '#636363', '#252525'] ,
'Greys_07' : ['#f7f7f7', '#d9d9d9', '#bdbdbd', '#969696', '#737373', '#525252', '#252525'] ,
'Greys_08' : ['#ffffff', '#f0f0f0', '#d9d9d9', '#bdbdbd', '#969696', '#737373', '#525252', '#252525'] ,
'Greys_09' : ['#ffffff', '#f0f0f0', '#d9d9d9', '#bdbdbd', '#969696', '#737373', '#525252', '#252525', '#000000'] ,
'OrRd_03' : ['#fee8c8', '#fdbb84', '#e34a33'] ,
'OrRd_04' : ['#fef0d9', '#fdcc8a', '#fc8d59', '#d7301f'] ,
'OrRd_05' : ['#fef0d9', '#fdcc8a', '#fc8d59', '#e34a33', '#b30000'] ,
'OrRd_06' : ['#fef0d9', '#fdd49e', '#fdbb84', '#fc8d59', '#e34a33', '#b30000'] ,
'OrRd_07' : ['#fef0d9', '#fdd49e', '#fdbb84', '#fc8d59', '#ef6548', '#d7301f', '#990000'] ,
'OrRd_08' : ['#fff7ec', '#fee8c8', '#fdd49e', '#fdbb84', '#fc8d59', '#ef6548', '#d7301f', '#990000'] ,
'OrRd_09' : ['#fff7ec', '#fee8c8', '#fdd49e', '#fdbb84', '#fc8d59', '#ef6548', '#d7301f', '#b30000', '#7f0000'] ,
'Oranges_03' : ['#fee6ce', '#fdae6b', '#e6550d'] ,
'Oranges_04' : ['#feedde', '#fdbe85', '#fd8d3c', '#d94701'] ,
'Oranges_05' : ['#feedde', '#fdbe85', '#fd8d3c', '#e6550d', '#a63603'] ,
'Oranges_06' : ['#feedde', '#fdd0a2', '#fdae6b', '#fd8d3c', '#e6550d', '#a63603'] ,
'Oranges_07' : ['#feedde', '#fdd0a2', '#fdae6b', '#fd8d3c', '#f16913', '#d94801', '#8c2d04'] ,
'Oranges_08' : ['#fff5eb', '#fee6ce', '#fdd0a2', '#fdae6b', '#fd8d3c', '#f16913', '#d94801', '#8c2d04'] ,
'Oranges_09' : ['#fff5eb', '#fee6ce', '#fdd0a2', '#fdae6b', '#fd8d3c', '#f16913', '#d94801', '#a63603', '#7f2704'] ,
'PRGn_03' : ['#af8dc3', '#f7f7f7', '#7fbf7b'] ,
'PRGn_04' : ['#7b3294', '#c2a5cf', '#a6dba0', '#008837'] ,
'PRGn_05' : ['#7b3294', '#c2a5cf', '#f7f7f7', '#a6dba0', '#008837'] ,
'PRGn_06' : ['#762a83', '#af8dc3', '#e7d4e8', '#d9f0d3', '#7fbf7b', '#1b7837'] ,
'PRGn_07' : ['#762a83', '#af8dc3', '#e7d4e8', '#f7f7f7', '#d9f0d3', '#7fbf7b', '#1b7837'] ,
'PRGn_08' : ['#762a83', '#9970ab', '#c2a5cf', '#e7d4e8', '#d9f0d3', '#a6dba0', '#5aae61', '#1b7837'] ,
'PRGn_09' : ['#762a83', '#9970ab', '#c2a5cf', '#e7d4e8', '#f7f7f7', '#d9f0d3', '#a6dba0', '#5aae61', '#1b7837'] ,
'PRGn_10' : ['#40004b', '#762a83', '#9970ab', '#c2a5cf', '#e7d4e8', '#d9f0d3', '#a6dba0', '#5aae61', '#1b7837', '#00441b'] ,
'PRGn_11' : ['#40004b', '#762a83', '#9970ab', '#c2a5cf', '#e7d4e8', '#f7f7f7', '#d9f0d3', '#a6dba0', '#5aae61', '#1b7837', '#00441b'] ,
'Paired_03' : ['#a6cee3', '#1f78b4', '#b2df8a'] ,
'Paired_04' : ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c'] ,
'Paired_05' : ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99'] ,
'Paired_06' : ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c'] ,
'Paired_07' : ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f'] ,
'Paired_08' : ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00'] ,
'Paired_09' : ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6'] ,
'Paired_10' : ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a'] ,
'Paired_11' : ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99'] ,
'Paired_12' : ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928'] ,
'Pastel1_03' : ['#fbb4ae', '#b3cde3', '#ccebc5'] ,
'Pastel1_04' : ['#fbb4ae', '#b3cde3', '#ccebc5', '#decbe4'] ,
'Pastel1_05' : ['#fbb4ae', '#b3cde3', '#ccebc5', '#decbe4', '#fed9a6'] ,
'Pastel1_06' : ['#fbb4ae', '#b3cde3', '#ccebc5', '#decbe4', '#fed9a6', '#ffffcc'] ,
'Pastel1_07' : ['#fbb4ae', '#b3cde3', '#ccebc5', '#decbe4', '#fed9a6', '#ffffcc', '#e5d8bd'] ,
'Pastel1_08' : ['#fbb4ae', '#b3cde3', '#ccebc5', '#decbe4', '#fed9a6', '#ffffcc', '#e5d8bd', '#fddaec'] ,
'Pastel1_09' : ['#fbb4ae', '#b3cde3', '#ccebc5', '#decbe4', '#fed9a6', '#ffffcc', '#e5d8bd', '#fddaec', '#f2f2f2'] ,
'Pastel2_03' : ['#b3e2cd', '#fdcdac', '#cbd5e8'] ,
'Pastel2_04' : ['#b3e2cd', '#fdcdac', '#cbd5e8', '#f4cae4'] ,
'Pastel2_05' : ['#b3e2cd', '#fdcdac', '#cbd5e8', '#f4cae4', '#e6f5c9'] ,
'Pastel2_06' : ['#b3e2cd', '#fdcdac', '#cbd5e8', '#f4cae4', '#e6f5c9', '#fff2ae'] ,
'Pastel2_07' : ['#b3e2cd', '#fdcdac', '#cbd5e8', '#f4cae4', '#e6f5c9', '#fff2ae', '#f1e2cc'] ,
'Pastel2_08' : ['#b3e2cd', '#fdcdac', '#cbd5e8', '#f4cae4', '#e6f5c9', '#fff2ae', '#f1e2cc', '#cccccc'] ,
'PiYG_03' : ['#e9a3c9', '#f7f7f7', '#a1d76a'] ,
'PiYG_04' : ['#d01c8b', '#f1b6da', '#b8e186', '#4dac26'] ,
'PiYG_05' : ['#d01c8b', '#f1b6da', '#f7f7f7', '#b8e186', '#4dac26'] ,
'PiYG_06' : ['#c51b7d', '#e9a3c9', '#fde0ef', '#e6f5d0', '#a1d76a', '#4d9221'] ,
'PiYG_07' : ['#c51b7d', '#e9a3c9', '#fde0ef', '#f7f7f7', '#e6f5d0', '#a1d76a', '#4d9221'] ,
'PiYG_08' : ['#c51b7d', '#de77ae', '#f1b6da', '#fde0ef', '#e6f5d0', '#b8e186', '#7fbc41', '#4d9221'] ,
'PiYG_09' : ['#c51b7d', '#de77ae', '#f1b6da', '#fde0ef', '#f7f7f7', '#e6f5d0', '#b8e186', '#7fbc41', '#4d9221'] ,
'PiYG_10' : ['#8e0152', '#c51b7d', '#de77ae', '#f1b6da', '#fde0ef', '#e6f5d0', '#b8e186', '#7fbc41', '#4d9221', '#276419'] ,
'PiYG_11' : ['#8e0152', '#c51b7d', '#de77ae', '#f1b6da', '#fde0ef', '#f7f7f7', '#e6f5d0', '#b8e186', '#7fbc41', '#4d9221', '#276419'] ,
'PuBuGn_03' : ['#ece2f0', '#a6bddb', '#1c9099'] ,
'PuBuGn_04' : ['#f6eff7', '#bdc9e1', '#67a9cf', '#02818a'] ,
'PuBuGn_05' : ['#f6eff7', '#bdc9e1', '#67a9cf', '#1c9099', '#016c59'] ,
'PuBuGn_06' : ['#f6eff7', '#d0d1e6', '#a6bddb', '#67a9cf', '#1c9099', '#016c59'] ,
'PuBuGn_07' : ['#f6eff7', '#d0d1e6', '#a6bddb', '#67a9cf', '#3690c0', '#02818a', '#016450'] ,
'PuBuGn_08' : ['#fff7fb', '#ece2f0', '#d0d1e6', '#a6bddb', '#67a9cf', '#3690c0', '#02818a', '#016450'] ,
'PuBuGn_09' : ['#fff7fb', '#ece2f0', '#d0d1e6', '#a6bddb', '#67a9cf', '#3690c0', '#02818a', '#016c59', '#014636'] ,
'PuBu_03' : ['#ece7f2', '#a6bddb', '#2b8cbe'] ,
'PuBu_04' : ['#f1eef6', '#bdc9e1', '#74a9cf', '#0570b0'] ,
'PuBu_05' : ['#f1eef6', '#bdc9e1', '#74a9cf', '#2b8cbe', '#045a8d'] ,
'PuBu_06' : ['#f1eef6', '#d0d1e6', '#a6bddb', '#74a9cf', '#2b8cbe', '#045a8d'] ,
'PuBu_07' : ['#f1eef6', '#d0d1e6', '#a6bddb', '#74a9cf', '#3690c0', '#0570b0', '#034e7b'] ,
'PuBu_08' : ['#fff7fb', '#ece7f2', '#d0d1e6', '#a6bddb', '#74a9cf', '#3690c0', '#0570b0', '#034e7b'] ,
'PuBu_09' : ['#fff7fb', '#ece7f2', '#d0d1e6', '#a6bddb', '#74a9cf', '#3690c0', '#0570b0', '#045a8d', '#023858'] ,
'PuOr_03' : ['#f1a340', '#f7f7f7', '#998ec3'] ,
'PuOr_04' : ['#e66101', '#fdb863', '#b2abd2', '#5e3c99'] ,
'PuOr_05' : ['#e66101', '#fdb863', '#f7f7f7', '#b2abd2', '#5e3c99'] ,
'PuOr_06' : ['#b35806', '#f1a340', '#fee0b6', '#d8daeb', '#998ec3', '#542788'] ,
'PuOr_07' : ['#b35806', '#f1a340', '#fee0b6', '#f7f7f7', '#d8daeb', '#998ec3', '#542788'] ,
'PuOr_08' : ['#b35806', '#e08214', '#fdb863', '#fee0b6', '#d8daeb', '#b2abd2', '#8073ac', '#542788'] ,
'PuOr_09' : ['#b35806', '#e08214', '#fdb863', '#fee0b6', '#f7f7f7', '#d8daeb', '#b2abd2', '#8073ac', '#542788'] ,
'PuOr_10' : ['#7f3b08', '#b35806', '#e08214', '#fdb863', '#fee0b6', '#d8daeb', '#b2abd2', '#8073ac', '#542788', '#2d004b'] ,
'PuOr_11' : ['#7f3b08', '#b35806', '#e08214', '#fdb863', '#fee0b6', '#f7f7f7', '#d8daeb', '#b2abd2', '#8073ac', '#542788', '#2d004b'] ,
'PuRd_03' : ['#e7e1ef', '#c994c7', '#dd1c77'] ,
'PuRd_04' : ['#f1eef6', '#d7b5d8', '#df65b0', '#ce1256'] ,
'PuRd_05' : ['#f1eef6', '#d7b5d8', '#df65b0', '#dd1c77', '#980043'] ,
'PuRd_06' : ['#f1eef6', '#d4b9da', '#c994c7', '#df65b0', '#dd1c77', '#980043'] ,
'PuRd_07' : ['#f1eef6', '#d4b9da', '#c994c7', '#df65b0', '#e7298a', '#ce1256', '#91003f'] ,
'PuRd_08' : ['#f7f4f9', '#e7e1ef', '#d4b9da', '#c994c7', '#df65b0', '#e7298a', '#ce1256', '#91003f'] ,
'PuRd_09' : ['#f7f4f9', '#e7e1ef', '#d4b9da', '#c994c7', '#df65b0', '#e7298a', '#ce1256', '#980043', '#67001f'] ,
'Purples_03' : ['#efedf5', '#bcbddc', '#756bb1'] ,
'Purples_04' : ['#f2f0f7', '#cbc9e2', '#9e9ac8', '#6a51a3'] ,
'Purples_05' : ['#f2f0f7', '#cbc9e2', '#9e9ac8', '#756bb1', '#54278f'] ,
'Purples_06' : ['#f2f0f7', '#dadaeb', '#bcbddc', '#9e9ac8', '#756bb1', '#54278f'] ,
'Purples_07' : ['#f2f0f7', '#dadaeb', '#bcbddc', '#9e9ac8', '#807dba', '#6a51a3', '#4a1486'] ,
'Purples_08' : ['#fcfbfd', '#efedf5', '#dadaeb', '#bcbddc', '#9e9ac8', '#807dba', '#6a51a3', '#4a1486'] ,
'Purples_09' : ['#fcfbfd', '#efedf5', '#dadaeb', '#bcbddc', '#9e9ac8', '#807dba', '#6a51a3', '#54278f', '#3f007d'] ,
'RdBu_03' : ['#ef8a62', '#f7f7f7', '#67a9cf'] ,
'RdBu_04' : ['#ca0020', '#f4a582', '#92c5de', '#0571b0'] ,
'RdBu_05' : ['#ca0020', '#f4a582', '#f7f7f7', '#92c5de', '#0571b0'] ,
'RdBu_06' : ['#b2182b', '#ef8a62', '#fddbc7', '#d1e5f0', '#67a9cf', '#2166ac'] ,
'RdBu_07' : ['#b2182b', '#ef8a62', '#fddbc7', '#f7f7f7', '#d1e5f0', '#67a9cf', '#2166ac'] ,
'RdBu_08' : ['#b2182b', '#d6604d', '#f4a582', '#fddbc7', '#d1e5f0', '#92c5de', '#4393c3', '#2166ac'] ,
'RdBu_09' : ['#b2182b', '#d6604d', '#f4a582', '#fddbc7', '#f7f7f7', '#d1e5f0', '#92c5de', '#4393c3', '#2166ac'] ,
'RdBu_10' : ['#67001f', '#b2182b', '#d6604d', '#f4a582', '#fddbc7', '#d1e5f0', '#92c5de', '#4393c3', '#2166ac', '#053061'] ,
'RdBu_11' : ['#67001f', '#b2182b', '#d6604d', '#f4a582', '#fddbc7', '#f7f7f7', '#d1e5f0', '#92c5de', '#4393c3', '#2166ac', '#053061'] ,
'RdGy_03' : ['#ef8a62', '#ffffff', '#999999'] ,
'RdGy_04' : ['#ca0020', '#f4a582', '#bababa', '#404040'] ,
'RdGy_05' : ['#ca0020', '#f4a582', '#ffffff', '#bababa', '#404040'] ,
'RdGy_06' : ['#b2182b', '#ef8a62', '#fddbc7', '#e0e0e0', '#999999', '#4d4d4d'] ,
'RdGy_07' : ['#b2182b', '#ef8a62', '#fddbc7', '#ffffff', '#e0e0e0', '#999999', '#4d4d4d'] ,
'RdGy_08' : ['#b2182b', '#d6604d', '#f4a582', '#fddbc7', '#e0e0e0', '#bababa', '#878787', '#4d4d4d'] ,
'RdGy_09' : ['#b2182b', '#d6604d', '#f4a582', '#fddbc7', '#ffffff', '#e0e0e0', '#bababa', '#878787', '#4d4d4d'] ,
'RdGy_10' : ['#67001f', '#b2182b', '#d6604d', '#f4a582', '#fddbc7', '#e0e0e0', '#bababa', '#878787', '#4d4d4d', '#1a1a1a'] ,
'RdGy_11' : ['#67001f', '#b2182b', '#d6604d', '#f4a582', '#fddbc7', '#ffffff', '#e0e0e0', '#bababa', '#878787', '#4d4d4d', '#1a1a1a'] ,
'RdPu_03' : ['#fde0dd', '#fa9fb5', '#c51b8a'] ,
'RdPu_04' : ['#feebe2', '#fbb4b9', '#f768a1', '#ae017e'] ,
'RdPu_05' : ['#feebe2', '#fbb4b9', '#f768a1', '#c51b8a', '#7a0177'] ,
'RdPu_06' : ['#feebe2', '#fcc5c0', '#fa9fb5', '#f768a1', '#c51b8a', '#7a0177'] ,
'RdPu_07' : ['#feebe2', '#fcc5c0', '#fa9fb5', '#f768a1', '#dd3497', '#ae017e', '#7a0177'] ,
'RdPu_08' : ['#fff7f3', '#fde0dd', '#fcc5c0', '#fa9fb5', '#f768a1', '#dd3497', '#ae017e', '#7a0177'] ,
'RdPu_09' : ['#fff7f3', '#fde0dd', '#fcc5c0', '#fa9fb5', '#f768a1', '#dd3497', '#ae017e', '#7a0177', '#49006a'] ,
'RdYlBu_03' : ['#fc8d59', '#ffffbf', '#91bfdb'] ,
'RdYlBu_04' : ['#d7191c', '#fdae61', '#abd9e9', '#2c7bb6'] ,
'RdYlBu_05' : ['#d7191c', '#fdae61', '#ffffbf', '#abd9e9', '#2c7bb6'] ,
'RdYlBu_06' : ['#d73027', '#fc8d59', '#fee090', '#e0f3f8', '#91bfdb', '#4575b4'] ,
'RdYlBu_07' : ['#d73027', '#fc8d59', '#fee090', '#ffffbf', '#e0f3f8', '#91bfdb', '#4575b4'] ,
'RdYlBu_08' : ['#d73027', '#f46d43', '#fdae61', '#fee090', '#e0f3f8', '#abd9e9', '#74add1', '#4575b4'] ,
'RdYlBu_09' : ['#d73027', '#f46d43', '#fdae61', '#fee090', '#ffffbf', '#e0f3f8', '#abd9e9', '#74add1', '#4575b4'] ,
'RdYlBu_10' : ['#a50026', '#d73027', '#f46d43', '#fdae61', '#fee090', '#e0f3f8', '#abd9e9', '#74add1', '#4575b4', '#313695'] ,
'RdYlBu_11' : ['#a50026', '#d73027', '#f46d43', '#fdae61', '#fee090', '#ffffbf', '#e0f3f8', '#abd9e9', '#74add1', '#4575b4', '#313695'] ,
'RdYlGn_03' : ['#fc8d59', '#ffffbf', '#91cf60'] ,
'RdYlGn_04' : ['#d7191c', '#fdae61', '#a6d96a', '#1a9641'] ,
'RdYlGn_05' : ['#d7191c', '#fdae61', '#ffffbf', '#a6d96a', '#1a9641'] ,
'RdYlGn_06' : ['#d73027', '#fc8d59', '#fee08b', '#d9ef8b', '#91cf60', '#1a9850'] ,
'RdYlGn_07' : ['#d73027', '#fc8d59', '#fee08b', '#ffffbf', '#d9ef8b', '#91cf60', '#1a9850'] ,
'RdYlGn_08' : ['#d73027', '#f46d43', '#fdae61', '#fee08b', '#d9ef8b', '#a6d96a', '#66bd63', '#1a9850'] ,
'RdYlGn_09' : ['#d73027', '#f46d43', '#fdae61', '#fee08b', '#ffffbf', '#d9ef8b', '#a6d96a', '#66bd63', '#1a9850'] ,
'RdYlGn_10' : ['#a50026', '#d73027', '#f46d43', '#fdae61', '#fee08b', '#d9ef8b', '#a6d96a', '#66bd63', '#1a9850', '#006837'] ,
'RdYlGn_11' : ['#a50026', '#d73027', '#f46d43', '#fdae61', '#fee08b', '#ffffbf', '#d9ef8b', '#a6d96a', '#66bd63', '#1a9850', '#006837'] ,
'Reds_03' : ['#fee0d2', '#fc9272', '#de2d26'] ,
'Reds_04' : ['#fee5d9', '#fcae91', '#fb6a4a', '#cb181d'] ,
'Reds_05' : ['#fee5d9', '#fcae91', '#fb6a4a', '#de2d26', '#a50f15'] ,
'Reds_06' : ['#fee5d9', '#fcbba1', '#fc9272', '#fb6a4a', '#de2d26', '#a50f15'] ,
'Reds_07' : ['#fee5d9', '#fcbba1', '#fc9272', '#fb6a4a', '#ef3b2c', '#cb181d', '#99000d'] ,
'Reds_08' : ['#fff5f0', '#fee0d2', '#fcbba1', '#fc9272', '#fb6a4a', '#ef3b2c', '#cb181d', '#99000d'] ,
'Reds_09' : ['#fff5f0', '#fee0d2', '#fcbba1', '#fc9272', '#fb6a4a', '#ef3b2c', '#cb181d', '#a50f15', '#67000d'] ,
'Set1_03' : ['#e41a1c', '#377eb8', '#4daf4a'] ,
'Set1_04' : ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3'] ,
'Set1_05' : ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00'] ,
'Set1_06' : ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33'] ,
'Set1_07' : ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628'] ,
'Set1_08' : ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf'] ,
'Set1_09' : ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf', '#999999'] ,
'Set2_03' : ['#66c2a5', '#fc8d62', '#8da0cb'] ,
'Set2_04' : ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3'] ,
'Set2_05' : ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854'] ,
'Set2_06' : ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f'] ,
'Set2_07' : ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494'] ,
'Set2_08' : ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3'] ,
'Set3_03' : ['#8dd3c7', '#ffffb3', '#bebada'] ,
'Set3_04' : ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072'] ,
'Set3_05' : ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3'] ,
'Set3_06' : ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462'] ,
'Set3_07' : ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462', '#b3de69'] ,
'Set3_08' : ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462', '#b3de69', '#fccde5'] ,
'Set3_09' : ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462', '#b3de69', '#fccde5', '#d9d9d9'] ,
'Set3_10' : ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462', '#b3de69', '#fccde5', '#d9d9d9', '#bc80bd'] ,
'Set3_11' : ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462', '#b3de69', '#fccde5', '#d9d9d9', '#bc80bd', '#ccebc5'] ,
'Set3_12' : ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462', '#b3de69', '#fccde5', '#d9d9d9', '#bc80bd', '#ccebc5', '#ffed6f'] ,
'Spectral_03' : ['#fc8d59', '#ffffbf', '#99d594'] ,
'Spectral_04' : ['#d7191c', '#fdae61', '#abdda4', '#2b83ba'] ,
'Spectral_05' : ['#d7191c', '#fdae61', '#ffffbf', '#abdda4', '#2b83ba'] ,
'Spectral_06' : ['#d53e4f', '#fc8d59', '#fee08b', '#e6f598', '#99d594', '#3288bd'] ,
'Spectral_07' : ['#d53e4f', '#fc8d59', '#fee08b', '#ffffbf', '#e6f598', '#99d594', '#3288bd'] ,
'Spectral_08' : ['#d53e4f', '#f46d43', '#fdae61', '#fee08b', '#e6f598', '#abdda4', '#66c2a5', '#3288bd'] ,
'Spectral_09' : ['#d53e4f', '#f46d43', '#fdae61', '#fee08b', '#ffffbf', '#e6f598', '#abdda4', '#66c2a5', '#3288bd'] ,
'Spectral_10' : ['#9e0142', '#d53e4f', '#f46d43', '#fdae61', '#fee08b', '#e6f598', '#abdda4', '#66c2a5', '#3288bd', '#5e4fa2'] ,
'Spectral_11' : ['#9e0142', '#d53e4f', '#f46d43', '#fdae61', '#fee08b', '#ffffbf', '#e6f598', '#abdda4', '#66c2a5', '#3288bd', '#5e4fa2'] ,
'YlGnBu_03' : ['#edf8b1', '#7fcdbb', '#2c7fb8'] ,
'YlGnBu_04' : ['#ffffcc', '#a1dab4', '#41b6c4', '#225ea8'] ,
'YlGnBu_05' : ['#ffffcc', '#a1dab4', '#41b6c4', '#2c7fb8', '#253494'] ,
'YlGnBu_06' : ['#ffffcc', '#c7e9b4', '#7fcdbb', '#41b6c4', '#2c7fb8', '#253494'] ,
'YlGnBu_07' : ['#ffffcc', '#c7e9b4', '#7fcdbb', '#41b6c4', '#1d91c0', '#225ea8', '#0c2c84'] ,
'YlGnBu_08' : ['#ffffd9', '#edf8b1', '#c7e9b4', '#7fcdbb', '#41b6c4', '#1d91c0', '#225ea8', '#0c2c84'] ,
'YlGnBu_09' : ['#ffffd9', '#edf8b1', '#c7e9b4', '#7fcdbb', '#41b6c4', '#1d91c0', '#225ea8', '#253494', '#081d58'] ,
'YlGn_03' : ['#f7fcb9', '#addd8e', '#31a354'] ,
'YlGn_04' : ['#ffffcc', '#c2e699', '#78c679', '#238443'] ,
'YlGn_05' : ['#ffffcc', '#c2e699', '#78c679', '#31a354', '#006837'] ,
'YlGn_06' : ['#ffffcc', '#d9f0a3', '#addd8e', '#78c679', '#31a354', '#006837'] ,
'YlGn_07' : ['#ffffcc', '#d9f0a3', '#addd8e', '#78c679', '#41ab5d', '#238443', '#005a32'] ,
'YlGn_08' : ['#ffffe5', '#f7fcb9', '#d9f0a3', '#addd8e', '#78c679', '#41ab5d', '#238443', '#005a32'] ,
'YlGn_09' : ['#ffffe5', '#f7fcb9', '#d9f0a3', '#addd8e', '#78c679', '#41ab5d', '#238443', '#006837', '#004529'] ,
'YlOrBr_03' : ['#fff7bc', '#fec44f', '#d95f0e'] ,
'YlOrBr_04' : ['#ffffd4', '#fed98e', '#fe9929', '#cc4c02'] ,
'YlOrBr_05' : ['#ffffd4', '#fed98e', '#fe9929', '#d95f0e', '#993404'] ,
'YlOrBr_06' : ['#ffffd4', '#fee391', '#fec44f', '#fe9929', '#d95f0e', '#993404'] ,
'YlOrBr_07' : ['#ffffd4', '#fee391', '#fec44f', '#fe9929', '#ec7014', '#cc4c02', '#8c2d04'] ,
'YlOrBr_08' : ['#ffffe5', '#fff7bc', '#fee391', '#fec44f', '#fe9929', '#ec7014', '#cc4c02', '#8c2d04'] ,
'YlOrBr_09' : ['#ffffe5', '#fff7bc', '#fee391', '#fec44f', '#fe9929', '#ec7014', '#cc4c02', '#993404', '#662506'] ,
'YlOrRd_03' : ['#ffeda0', '#feb24c', '#f03b20'] ,
'YlOrRd_04' : ['#ffffb2', '#fecc5c', '#fd8d3c', '#e31a1c'] ,
'YlOrRd_05' : ['#ffffb2', '#fecc5c', '#fd8d3c', '#f03b20', '#bd0026'] ,
'YlOrRd_06' : ['#ffffb2', '#fed976', '#feb24c', '#fd8d3c', '#f03b20', '#bd0026'] ,
'YlOrRd_07' : ['#ffffb2', '#fed976', '#feb24c', '#fd8d3c', '#fc4e2a', '#e31a1c', '#b10026'] ,
'YlOrRd_08' : ['#ffffcc', '#ffeda0', '#fed976', '#feb24c', '#fd8d3c', '#fc4e2a', '#e31a1c', '#b10026'] ,
'YlOrRd_09' : ['#ffffcc', '#ffeda0', '#fed976', '#feb24c', '#fd8d3c', '#fc4e2a', '#e31a1c', '#bd0026', '#800026'] ,
}



'''
-------------------------------------------
常用的color集合
包含颜色按color_names的顺序：
前12个是Paired_12, 再Pastel1_08，再Dark2_08
'''


def create_colorset(color_names):
    color_set2 = []
    for each in color_names:
        if  each in ColorScheme.keys():
            color_temp = [color for color in ColorScheme[each] if color not in color_set2]
            color_set2.extend(color_temp)
    return color_set2

color_names = ['Paired_12','Pastel1_08','Dark2_08','GnBu_09']
# ColorSet = create_colorset()

ColorSet = ['#33a02c', '#fb9a99','#e31a1c', '#fdbf6f', '#ff7f00',
            '#cab2d6', '#6a3d9a','#a6cee3', '#1f78b4', '#b2df8a',
            '#ffff99', '#b15928',

            '#fbb4ae', '#b3cde3', '#ccebc5',
            '#decbe4', '#fed9a6', '#ffffcc', '#e5d8bd', '#fddaec',
            '#1b9e77', '#d95f02', '#7570b3', '#e7298a',

            '#66a61e', '#e6ab02', '#a6761d', '#666666',
            '#e0f3db', '#a8ddb5', '#7bccc4', '#4eb3d3', '#2b8cbe',
            '#0868ac', '#084081']


common_discrete_color_num = len(ColorSet)


def get_common_discrete_colors():
    return ColorSet


def is_support_cmap(cmap):
    return cmap in ColorScheme.keys()

def get_colors(cmap):
    if(isinstance(cmap,str)):
        colors_i = ColorScheme.get(cmap)
        if(colors_i is None):
            return []
        else:
            return colors_i
    else:
        colors_ = []
        for each in cmap:
            colors_i = ColorScheme.get(cmap)
            if(colors_i is not None):
                colors_.extend(colors_i)
        return colors_


def find_unsupport_cmaps(cmap):
    unsupport_camps = []

    if(isinstance(cmap,str)):
        if(not is_support_cmap(cmap)):
            unsupport_camps.append(cmap)
    else:
        for each in cmap:
            if(not is_support_cmap(cmap)):
                unsupport_camps.append(each)

    return unsupport_camps 



