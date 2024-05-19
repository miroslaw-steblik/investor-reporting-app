# The colorlover library provides several color scales that you can use. Here are some of them:

#Example:
['seq']['Greys']


"""
Sequential ('seq'): These color scales are used for ordered data that progresses from low to high. 
They are often single-hue or multi-hue color scales. Lightness steps dominate the look of these schemes, 
with light colors for low data values to dark colors for high data values. 
"""

['seq']

'Greys'
'YlGnBu'
'Greens'
'YlOrRd'
'Bluered'
'RdBu'
'Reds'
'Blues'
'Picnic'
'Rainbow'
'Portland'
'Jet'
'Hot'
'Blackbody'
'Earth'
'Electric'
'Viridis'
'Cividis'
'Inferno'

"""
Diverging ('div'): These color scales are used for data values deviating around a middle value, 
both low to high and high to low. They are often used when the data has a critical midpoint like zero. 
These scales use two contrasting colors that diverge from a light color at the midpoint. 
"""

['div']

'Spectral'
'RdBu'
'RdYlBu'
'RdYlGn'
'RdGy'
'PRGn'
'PiYG'
'BrBG'
'PuOr'


"""
Qualitative ('qual'): These color scales are used for categorical data. 
They consist of distinct colors that do not imply any ordering or relationship between categories. 
Each color represents a different category. Examples include 'Set1', 'Set2', 'Set3', 'Paired', etc.
"""
['qual']

'Set1'
'Set2'
'Set3'
'Dark2'
'Paired'
'Pastel1'
'Pastel2'
'Accent'
'Vega10'