#usecols = [26, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 50, 51, 52, 58, 60, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 76, 87, 88, 95, 96, 97, 103, 104, 107, 108, 109, 116, 117, 127, 128, 129, 131, 132, 133, 134, 135, 136, 137, 138, 140, 141, 142, 144, 146, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 192, 193, 194, 196, 198, 199, 200, 201, 202, 203, 204, 205, 214, 215]
#usecols = [248, 251, 253, 266, 267, 268, 269, 270, 271, 276, 277, 291, 292, 295, 296, 302, 303, 304]
usecols = [26, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 50, 51, 52, 58, 60, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 76, 87, 88, 95, 96, 97, 103, 104, 107, 108, 109, 116, 117, 127, 128, 129, 131, 132, 133, 134, 135, 136, 137, 138, 140, 141, 142, 144, 146, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 192, 193, 194, 196, 198, 199, 200, 201, 202, 203, 204, 205, 214, 215, 248, 251, 253, 266, 267, 268, 269, 270, 271, 276, 277, 291, 292, 295, 296, 302, 303, 304] # False : 248, 251, 253, 266, 267, 268, 269, 270, 271, 276, 277, 291, 292, 295, 296, 302, 303, 304 (last 18 elems)


print('[', end='')
for col in usecols:
    print(f'{col+1}, ', end='')
print(']', end='')
