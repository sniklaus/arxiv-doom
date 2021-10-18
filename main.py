#!/usr/bin/env python

import cv2
import feedparser
import json
import math
import numpy
import os
import pdf2image # sudo apt-get install poppler-utils
import tqdm
import urllib
import urllib.request


##########################################################


def get_composite(npySrc, npyTarget, intX, intY):
    npyAlpha = npySrc[:, :, 3:4].astype(numpy.float32) * (1.0 / 255.0)

    npyComposed = ((npyTarget[intY:(intY + npySrc.shape[0]), intX:(intX + npySrc.shape[1]), :] * (1.0 - npyAlpha)) + npySrc).clip(0.0, 255.0).round().astype(numpy.uint8)

    npyOut = npyTarget.copy(); npyOut[intY:(intY + npySrc.shape[0]), intX:(intX + npySrc.shape[1]), :] = npyComposed

    return npyOut
# end


def get_rotation(intWidth, intHeight, fltFocal, fltRotx, fltRoty, fltRotz):
    # https://stackoverflow.com/a/19110462

    fltCosx, fltSinx = math.cos(fltRotx), math.sin(fltRotx)
    fltCosy, fltSiny = math.cos(fltRoty), math.sin(fltRoty)
    fltCosz, fltSinz = math.cos(fltRotz), math.sin(fltRotz)

    fltRot = [
        [fltCosz * fltCosy, (fltCosz * fltSiny * fltSinx) - (fltSinz * fltCosx)],
        [fltSinz * fltCosy, (fltSinz * fltSiny * fltSinx) + (fltCosz * fltCosx)],
        [-fltSiny, fltCosy * fltSinx]
    ]

    fltDst = []

    for fltPt in [[-0.5 * intWidth, -0.5 * intHeight], [0.5 * intWidth, -0.5 * intHeight], [-0.5 * intWidth, 0.5 * intHeight], [0.5 * intWidth, 0.5 * intHeight]]:
        fltPtx = (fltPt[0] * fltRot[0][0]) + (fltPt[1] * fltRot[0][1])
        fltPty = (fltPt[0] * fltRot[1][0]) + (fltPt[1] * fltRot[1][1])
        fltPtz = (fltPt[0] * fltRot[2][0]) + (fltPt[1] * fltRot[2][1])

        fltDst.append([
            (0.5 * intWidth) + ((fltPtx * fltFocal) / (fltPtz + fltFocal)),
            (0.5 * intHeight) + ((fltPty * fltFocal) / (fltPtz + fltFocal))
        ])
    # end

    npySrc = numpy.array([[0.0, 0.0], [intWidth, 0.0], [0.0, intHeight], [intWidth, intHeight]], numpy.float32)
    npyDst = numpy.array(fltDst, numpy.float32)

    return cv2.getPerspectiveTransform(src=npySrc, dst=npyDst)
# end


def get_scale(intWidth, intHeight, fltScale):
    return numpy.array([[fltScale, 0.0, 0.5 * (1.0 - fltScale) * intWidth], [0.0, fltScale, 0.5 * (1.0 - fltScale) * intHeight], [0.0, 0.0, 1.0]], numpy.float32)
# end


def get_spritesheet(npyFront, npyBack):
    assert(npyFront.shape == tuple([550, 425, 3]) and npyFront.dtype == numpy.uint8)
    assert(npyBack.shape == tuple([550, 425, 3]) and npyBack.dtype == numpy.uint8)

    npyFront = cv2.resize(src=npyFront, dsize=None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    npyBack = cv2.resize(src=npyBack, dsize=None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)

    intWidth = npyFront.shape[1] and npyBack.shape[1]
    intHeight = npyFront.shape[0] and npyBack.shape[0]

    npyFront = numpy.concatenate([npyFront, numpy.full([intHeight, intWidth, 1], 255, numpy.uint8)], 2)
    npyBack = numpy.concatenate([npyBack, numpy.full([intHeight, intWidth, 1], 255, numpy.uint8)], 2)

    npyRejected = cv2.imread(filename=os.path.dirname(os.path.abspath(__file__)) + '/images/rejected.png', flags=-1)
    npyRejected = cv2.resize(src=npyRejected, dsize=None, fx=intWidth / npyRejected.shape[1] * 0.9, fy=intWidth / npyRejected.shape[1] * 0.9, interpolation=cv2.INTER_AREA)
    npyRejected = get_composite(npyRejected, npyFront, int(round((0.5 * intWidth) - (0.5 * npyRejected.shape[1]))), int(round((0.3 * intHeight) - (0.5 * npyRejected.shape[0]))))

    intPadl = int(math.floor(0.5 * (intHeight - intWidth)))
    intPadr = int(math.ceil(0.5 * (intHeight - intWidth)))
    npyFront = numpy.pad(npyFront, [(1, 1), (intPadl, intPadr), (0, 0)], 'constant', constant_values=[((255, 255, 255, 0), (255, 255, 255, 0)), ((255, 255, 255, 0), (255, 255, 255, 0)), (0, 0)])
    npyBack = numpy.pad(npyBack, [(1, 1), (intPadl, intPadr), (0, 0)], 'constant', constant_values=[((255, 255, 255, 0), (255, 255, 255, 0)), ((255, 255, 255, 0), (255, 255, 255, 0)), (0, 0)])
    npyRejected = numpy.pad(npyRejected, [(1, 1), (intPadl, intPadr), (0, 0)], 'constant', constant_values=[((255, 255, 255, 0), (255, 255, 255, 0)), ((255, 255, 255, 0), (255, 255, 255, 0)), (0, 0)])

    intWidth = npyFront.shape[1] and npyBack.shape[1]
    intHeight = npyFront.shape[0] and npyBack.shape[0]

    npyPerspectives = [numpy.concatenate([numpy.full([intHeight, intWidth, 3], 255, numpy.uint8), numpy.full([intHeight, intWidth, 1], 0, numpy.uint8)], 2) for intSprite in range(18)]
    npyDeaths = [numpy.concatenate([numpy.full([intHeight, intWidth, 3], 255, numpy.uint8), numpy.full([intHeight, intWidth, 1], 0, numpy.uint8)], 2) for intSprite in range(18)]

    for intSprite, fltRotation in enumerate(numpy.linspace(0.0, (17 / 18) * 2.0 * math.pi, 18).tolist()):
        npyHomo = get_rotation(intWidth, intHeight, 2.0 * intWidth, 0.0, fltRotation, 0.0)

        for fltScale in numpy.linspace(1.0, 0.7, 100).tolist():
            if fltRotation < 0.5 * math.pi or fltRotation >= 1.5 * math.pi:
                npyRotation = cv2.warpPerspective(src=npyFront, M=numpy.matmul(get_scale(intWidth, intHeight, fltScale), npyHomo), dsize=(intWidth, intHeight), flags=cv2.INTER_LINEAR)

            elif fltRotation >= 0.5 * math.pi and fltRotation < 1.5 * math.pi:
                npyRotation = cv2.warpPerspective(src=npyBack, M=numpy.matmul(get_scale(intWidth, intHeight, fltScale), npyHomo), dsize=(intWidth, intHeight), flags=cv2.INTER_LINEAR)

            # end

            if npyRotation[0, :, 3].sum().item() == 0.0:
                break
            # end
        # end

        npyPerspectives[intSprite] = npyRotation
    # end

    for intSprite, fltRotation in enumerate(numpy.linspace(0.0, -0.41 * math.pi, 6).tolist()):
        npyHomo = get_rotation(intWidth, intHeight, 2.0 * intWidth, fltRotation, 0.0, 0.0)

        for fltScale in numpy.linspace(1.0, 0.7, 100).tolist():
            npyRotation = cv2.warpPerspective(src=npyRejected, M=numpy.matmul(get_scale(intWidth, intHeight, fltScale), npyHomo), dsize=(intWidth, intHeight), flags=cv2.INTER_LINEAR)

            if npyRotation[0, :, :].sum().item() == 0.0:
                break
            # end
        # end

        for intY in range(intHeight - 1, 0, -1):
            if npyRotation[intY, :, 3].sum() != 0:
                intCrop = intHeight - intY - 1

                npyRotation = numpy.concatenate([numpy.concatenate([numpy.full([intHeight - intY, intWidth, 3], 255, numpy.uint8), numpy.full([intHeight - intY, intWidth, 1], 0, numpy.uint8)], 2), npyRotation[:intY, :, :]], 0)

                break
            # end
        # end

        npyDeaths[intSprite] = npyRotation
    # end

    return numpy.concatenate([numpy.concatenate(npyPerspectives, 1), numpy.concatenate(npyDeaths, 1)], 0)
# end


##########################################################


if __name__ == '__main__':
    strMine = [
        '1703.07514',
        '1708.01692',
        '1803.10967',
        '1909.05483',
        '2003.05534',
        '2010.00702',
        '2011.01280'
    ]

    objPapers = []
    objPapers += feedparser.parse('http://export.arxiv.org/api/query?id_list=' + str(',').join(strMine)).entries
    objPapers += feedparser.parse('http://export.arxiv.org/api/query?search_query=cat:cs.CV&start=0&max_results=100&sortBy=lastUpdatedDate&sortOrder=descending').entries

    objExpose = []

    for objPaper in tqdm.tqdm(objPapers):
        try:
            objMeta = {
                'strIdent': objPaper.id.split('/')[-1].split('v')[0],
                'strTitle': str(' ').join([strChunk.strip() for strChunk in objPaper.title.split('\n')]),
                'strAuthors': str(', ').join([objAuthor.name for objAuthor in objPaper.authors]),
                'strSummary': str(' ').join([strChunk.strip() for strChunk in objPaper.summary.split('\n')]),
                'strPdf': [objLink.href for objLink in objPaper.links if 'title' in objLink.keys() and objLink.title == 'pdf'][0]
            }

            if os.path.exists(os.path.dirname(os.path.abspath(__file__)) + '/papers/' + objMeta['strIdent'] + '.json') == False:
                npyPages = [numpy.array(objPage)[:, :, ::-1] for objPage in pdf2image.convert_from_bytes(pdf_file=urllib.request.urlopen(objMeta['strPdf']).read(), dpi=50)[:2]]
                npySpritesheet = get_spritesheet(npyPages[0], npyPages[1])

                with open(os.path.dirname(os.path.abspath(__file__)) + '/papers/' + objMeta['strIdent'] + '.json', 'w') as objFile:
                    objFile.write(json.dumps(objMeta))
                # end

                cv2.imwrite(filename=os.path.dirname(os.path.abspath(__file__)) + '/papers/' + objMeta['strIdent'] + '.png', img=npySpritesheet, params=[cv2.IMWRITE_PNG_COMPRESSION, 9])
            # end

            objExpose.append(objMeta)
        except:
            pass
        # end
    # end
# end

with open(os.path.dirname(os.path.abspath(__file__)) + '/main.json', 'w') as objFile:
    objFile.write(json.dumps(objExpose))
# end
