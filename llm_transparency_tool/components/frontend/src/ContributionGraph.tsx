/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

import {
    ComponentProps,
    Streamlit,
    withStreamlitConnection,
} from 'streamlit-component-lib'
import React, { useEffect, useMemo, useRef, useState } from 'react';
import * as d3 from 'd3';

import {
    Label,
    Point,
} from './common';
import './LlmViewer.css';

const renderParams = {
    cellH: 32,
    minCellW: 12,
    maxCellW: 32,
    minAttnSize: 6,
    maxAttnSize: 8,
    minFfnSize: 4,
    maxFfnSize: 6,
    minTokenSelectorSize: 8,
    maxTokenSelectorSize: 16,
    layerCornerRadius: 6,
    topPad: 12,
    leftPad: 56,
    rightPad: 48,
    minViewportW: 320,
    labelSpacing: 72,
}

interface Cell {
    layer: number
    token: number
}

enum CellItem {
    AfterAttn = 'after_attn',
    AfterFfn = 'after_ffn',
    Ffn = 'ffn',
    Original = 'original',  // They will only be at level = 0
}

interface Node {
    cell: Cell | null
    item: CellItem | null
}

interface NodeProps {
    node: Node
    pos: Point
    isActive: boolean
}

interface EdgeRaw {
    weight: number
    source: string
    target: string
}

interface Edge {
    weight: number
    from: Node
    to: Node
    fromPos: Point
    toPos: Point
    isSelectable: boolean
    isFfn: boolean
}

interface Selection {
    node: Node | null
    edge: Edge | null
}

function clamp(value: number, min: number, max: number) {
    return Math.min(Math.max(value, min), max)
}

function tokenPointerPolygon(origin: Point, size: number) {
    const r = size / 2
    const dy = r / 2
    const dx = r * Math.sqrt(3.0) / 2
    // Draw an arrow looking down
    return [
        [origin.x, origin.y + r],
        [origin.x + dx, origin.y - dy],
        [origin.x - dx, origin.y - dy],
    ].toString()
}

function isSameCell(cell1: Cell | null, cell2: Cell | null) {
    if (cell1 == null || cell2 == null) {
        return false
    }
    return cell1.layer === cell2.layer && cell1.token === cell2.token
}

function isSameNode(node1: Node | null, node2: Node | null) {
    if (node1 === null || node2 === null) {
        return false
    }
    return isSameCell(node1.cell, node2.cell)
        && node1.item === node2.item;
}

function isSameEdge(edge1: Edge | null, edge2: Edge | null) {
    if (edge1 === null || edge2 === null) {
        return false
    }
    return isSameNode(edge1.from, edge2.from) && isSameNode(edge1.to, edge2.to);
}

function nodeFromString(name: string) {
    const match = name.match(/([AIMX])(\d+)_(\d+)/)
    if (match == null) {
        return {
            cell: null,
            item: null,
        }
    }
    const [, type, layerStr, tokenStr] = match
    const layer = +layerStr
    const token = +tokenStr

    const typeToCellItem = new Map<string, CellItem>([
        ['A', CellItem.AfterAttn],
        ['I', CellItem.AfterFfn],
        ['M', CellItem.Ffn],
        ['X', CellItem.Original],
    ])
    return {
        cell: {
            layer: layer,
            token: token,
        },
        item: typeToCellItem.get(type) ?? null,
    }
}

function isValidNode(node: Node, nLayers: number, nTokens: number) {
    if (node.cell === null) {
        return true
    }
    return node.cell.layer < nLayers && node.cell.token < nTokens
}

function isValidSelection(selection: Selection, nLayers: number, nTokens: number) {
    if (selection.node !== null) {
        return isValidNode(selection.node, nLayers, nTokens)
    }
    if (selection.edge !== null) {
        return isValidNode(selection.edge.from, nLayers, nTokens) &&
            isValidNode(selection.edge.to, nLayers, nTokens)
    }
    return true
}

const ContributionGraph = ({ args }: ComponentProps) => {
    const modelInfo = args['model_info']
    const tokens = args['tokens']
    const edgesRaw: EdgeRaw[][] = args['edges_per_token']

    const nLayers = modelInfo === null ? 0 : modelInfo.n_layers
    const nTokens = tokens === null ? 0 : tokens.length

    const [selection, setSelection] = useState<Selection>({
        node: null,
        edge: null,
    })
    var curSelection = selection
    if (!isValidSelection(selection, nLayers, nTokens)) {
        curSelection = {
            node: null,
            edge: null,
        }
        setSelection(curSelection)
        Streamlit.setComponentValue(curSelection)
    }

    const [startToken, setStartToken] = useState<number>(0)
    // We have startToken state var, but it won't be updated till next render, so use
    // this var in the current render.
    var curStartToken = startToken
    if (nTokens === 0) {
        curStartToken = 0
    } else if (startToken >= nTokens) {
        curStartToken = nTokens - 1
        setStartToken(curStartToken)
    }

    const handleRepresentationClick = (node: Node) => {
        const newSelection: Selection = {
            node: node,
            edge: null,
        }
        setSelection(newSelection)
        Streamlit.setComponentValue(newSelection)
    }

    const handleEdgeClick = (edge: Edge) => {
        if (!edge.isSelectable) {
            return
        }
        const newSelection: Selection = {
            node: edge.to,
            edge: edge,
        }
        setSelection(newSelection)
        Streamlit.setComponentValue(newSelection)
    }

    const handleTokenClick = (t: number) => {
        setStartToken(t)
    }

    const containerRef = useRef<HTMLDivElement | null>(null);
    const svgRef = useRef<SVGSVGElement | null>(null);
    const [viewportWidth, setViewportWidth] = useState<number>(renderParams.minViewportW)

    useEffect(() => {
        const container = containerRef.current
        if (container === null) {
            return
        }

        const updateWidth = () => {
            setViewportWidth(Math.max(container.clientWidth, renderParams.minViewportW))
        }

        updateWidth()

        const observer = new ResizeObserver(() => {
            updateWidth()
        })
        observer.observe(container)

        return () => {
            observer.disconnect()
        }
    }, [])

    const geometry = useMemo(() => {
        const tokenSlots = Math.max(nTokens, 1)
        const usableWidth = Math.max(
            viewportWidth - renderParams.leftPad - renderParams.rightPad,
            renderParams.minCellW * tokenSlots,
        )
        const cellW = clamp(
            Math.floor(usableWidth / tokenSlots),
            renderParams.minCellW,
            renderParams.maxCellW,
        )
        const totalW = renderParams.leftPad + renderParams.rightPad + cellW * tokenSlots
        return {
            cellH: renderParams.cellH,
            cellW,
            attnSize: clamp(cellW * 0.45, renderParams.minAttnSize, renderParams.maxAttnSize),
            ffnSize: clamp(cellW * 0.3, renderParams.minFfnSize, renderParams.maxFfnSize),
            tokenSelectorSize: clamp(
                cellW * 0.7,
                renderParams.minTokenSelectorSize,
                renderParams.maxTokenSelectorSize,
            ),
            leftPad: renderParams.leftPad,
            rightPad: renderParams.rightPad,
            totalW,
        }
    }, [nTokens, viewportWidth])

    const isAudioTimeline = useMemo(() => {
        if (modelInfo?.name === 'mt2') {
            return true
        }
        if (!tokens || tokens.length <= 2) {
            return false
        }
        return false
    }, [modelInfo, tokens])

    const tokenLabelStep = useMemo(() => {
        if (isAudioTimeline) {
            return Math.max(1, Math.ceil(renderParams.labelSpacing / geometry.cellW))
        }

        const maxLabelCount = Math.max(
            1,
            Math.floor(
                Math.max(viewportWidth - geometry.leftPad, renderParams.labelSpacing) /
                renderParams.labelSpacing,
            ),
        )

        return Math.max(1, Math.ceil(Math.max(nTokens, 1) / maxLabelCount))
    }, [geometry.cellW, geometry.leftPad, isAudioTimeline, nTokens, viewportWidth])

    const [xScale, yScale] = useMemo(() => {
        const tokenSlots = Math.max(nTokens, 1)
        const x = d3.scaleLinear()
            .domain([0, tokenSlots])
            .range([geometry.leftPad, geometry.leftPad + geometry.cellW * tokenSlots])
        const y = d3.scaleLinear()
            .domain([-1, nLayers + 1.5])
            .range([geometry.cellH * (nLayers + 3.5) + renderParams.topPad, renderParams.topPad])
        return [x, y]
    }, [geometry.cellH, geometry.cellW, geometry.leftPad, nLayers, nTokens])

    const cells = useMemo(() => {
        let result: Cell[] = []
        for (let l = 0; l < nLayers; l++) {
            for (let t = 0; t < nTokens; t++) {
                result.push({
                    layer: l,
                    token: t,
                })
            }
        }
        return result
    }, [nLayers, nTokens])

    const nodeCoords = useMemo(() => {
        let result = new Map<string, Point>()
        const w = geometry.cellW
        const h = geometry.cellH
        for (var cell of cells) {
            const cx = xScale(cell.token + 0.5)
            const cy = yScale(cell.layer - 0.5)
            result.set(
                JSON.stringify({ cell: cell, item: CellItem.AfterAttn }),
                { x: cx, y: cy + h / 4 },
            )
            result.set(
                JSON.stringify({ cell: cell, item: CellItem.AfterFfn }),
                { x: cx, y: cy - h / 4 },
            )
            result.set(
                JSON.stringify({ cell: cell, item: CellItem.Ffn }),
                { x: cx + 5 * w / 16, y: cy },
            )
        }
        for (let t = 0; t < nTokens; t++) {
            cell = {
                layer: 0,
                token: t,
            }
            const cx = xScale(cell.token + 0.5)
            const cy = yScale(cell.layer - 1.0)
            result.set(
                JSON.stringify({ cell: cell, item: CellItem.Original }),
                { x: cx, y: cy + h / 4 },
            )
        }
        return result
    }, [cells, geometry.cellH, geometry.cellW, nTokens, xScale, yScale])

    const edges: Edge[][] = useMemo(() => {
        let result = []
        for (var edgeList of edgesRaw) {
            let edgesPerStartToken = []
            for (var edge of edgeList) {
                const u = nodeFromString(edge.source)
                const v = nodeFromString(edge.target)
                var isSelectable = (
                    u.cell !== null && v.cell !== null && v.item === CellItem.AfterAttn
                )
                var isFfn = (
                    u.cell !== null && v.cell !== null && (
                        u.item === CellItem.Ffn || v.item === CellItem.Ffn
                    )
                )
                edgesPerStartToken.push({
                    weight: edge.weight,
                    from: u,
                    to: v,
                    fromPos: nodeCoords.get(JSON.stringify(u)) ?? { 'x': 0, 'y': 0 },
                    toPos: nodeCoords.get(JSON.stringify(v)) ?? { 'x': 0, 'y': 0 },
                    isSelectable: isSelectable,
                    isFfn: isFfn,
                })
            }
            result.push(edgesPerStartToken)
        }
        return result
    }, [edgesRaw, nodeCoords])

    const currentEdges = useMemo(() => edges[curStartToken] ?? [], [curStartToken, edges])

    const activeNodes = useMemo(() => {
        let result = new Set<string>()
        for (var edge of currentEdges) {
            const u = JSON.stringify(edge.from)
            const v = JSON.stringify(edge.to)
            result.add(u)
            result.add(v)
        }
        return result
    }, [currentEdges])

    const nodeProps = useMemo(() => {
        let result: Array<NodeProps> = []
        nodeCoords.forEach((p: Point, node: string) => {
            result.push({
                node: JSON.parse(node),
                pos: p,
                isActive: activeNodes.has(node),
            })
        })
        return result
    }, [nodeCoords, activeNodes])

    const tokenLabels: Label[] = useMemo(() => {
        if (!tokens) {
            return []
        }
        return tokens.flatMap((s: string, i: number) => {
            if (isAudioTimeline) {
                if (i < 2) {
                    return []
                }
                const audioTokenIndex = i - 2
                const shouldShow = (
                    audioTokenIndex % tokenLabelStep === 0 ||
                    i === nTokens - 1
                )
                if (!shouldShow) {
                    return []
                }
                return [{
                    text: s.replace(/ /g, '·'),
                    pos: {
                        x: xScale(i + 0.5),
                        y: yScale(-2.25),
                    },
                }]
            }

            const isSpecialToken = i < 2 || i === nTokens - 1
            const shouldShow = isSpecialToken || (
                i % tokenLabelStep === 0
            )
            if (!shouldShow) {
                return []
            }
            return [{
                text: s.replace(/ /g, '·'),
                pos: {
                    x: xScale(i + 0.5),
                    y: yScale(nLayers + 1.2),
                },
            }]
        })
    }, [isAudioTimeline, nLayers, nTokens, tokenLabelStep, tokens, xScale, yScale])

    const audioAxisY = useMemo(() => yScale(-1.65), [yScale])

    const layerLabels: Label[] = useMemo(() => {
        return Array.from(Array(nLayers).keys()).map(i => ({
            text: 'L' + i,
            pos: {
                x: xScale(-0.25),
                y: yScale(i - 0.5),
            },
        }))
    }, [nLayers, xScale, yScale])

    const tokenSelectors: Array<[number, Point]> = useMemo(() => {
        return Array.from(Array(nTokens).keys()).map(i => ([
            i,
            {
                x: xScale(i + 0.5),
                y: yScale(nLayers - 0.5),
            }
        ]))
    }, [nTokens, nLayers, xScale, yScale])

    const totalW = geometry.totalW
    const totalH = yScale(-4)
    const needsHorizontalScroll = totalW > viewportWidth

    useEffect(() => {
        Streamlit.setFrameHeight(totalH + (needsHorizontalScroll ? 20 : 0))
    }, [needsHorizontalScroll, totalH])

    const colorScale = d3.scaleLinear(
        [0.0, 0.5, 1.0],
        ['#9eba66', 'darkolivegreen', 'darkolivegreen']
    )
    const ffnEdgeColorScale = d3.scaleLinear(
        [0.0, 0.5, 1.0],
        ['orchid', 'purple', 'purple']
    )
    const edgeWidthScale = d3.scaleLinear([0.0, 0.5, 1.0], [2.0, 3.0, 3.0])

    useEffect(() => {
        const getNodeStyle = (p: NodeProps, type: string) => {
            if (isSameNode(p.node, curSelection.node)) {
                return 'selectable-item selection'
            }
            if (p.isActive) {
                return 'selectable-item active-' + type + '-node'
            }
            return 'selectable-item inactive-node'
        }

        const svg = d3.select(svgRef.current)
        svg.selectAll('*').remove()

        svg
            .selectAll('layers')
            .data(Array.from(Array(nLayers).keys()).filter((x) => x % 2 === 1))
            .enter()
            .append('rect')
            .attr('class', 'layer-highlight')
            .attr('x', geometry.leftPad - geometry.cellW * 0.75)
            .attr('y', (layer) => yScale(layer))
            .attr('width', Math.max(nTokens, 1) * geometry.cellW + geometry.cellW * 0.5)
            .attr('height', (layer) => yScale(layer) - yScale(layer + 1))
            .attr('rx', renderParams.layerCornerRadius)

        svg
            .selectAll('edges')
            .data(currentEdges)
            .enter()
            .append('line')
            .style('stroke', (edge: Edge) => {
                if (isSameEdge(edge, curSelection.edge)) {
                    return 'orange'
                }
                if (edge.isFfn) {
                    return ffnEdgeColorScale(edge.weight)
                }
                return colorScale(edge.weight)
            })
            .attr('class', (edge: Edge) => edge.isSelectable ? 'selectable-edge' : '')
            .style('stroke-width', (edge: Edge) => edgeWidthScale(edge.weight))
            .attr('x1', (edge: Edge) => edge.fromPos.x)
            .attr('y1', (edge: Edge) => edge.fromPos.y)
            .attr('x2', (edge: Edge) => edge.toPos.x)
            .attr('y2', (edge: Edge) => edge.toPos.y)
            .on('click', (event: PointerEvent, edge) => {
                handleEdgeClick(edge)
            })

        svg
            .selectAll('residual')
            .data(nodeProps)
            .enter()
            .filter((p) => {
                return p.node.item === CellItem.AfterAttn
                    || p.node.item === CellItem.AfterFfn
            })
            .append('circle')
            .attr('class', (p) => getNodeStyle(p, 'residual'))
            .attr('cx', (p) => p.pos.x)
            .attr('cy', (p) => p.pos.y)
            .attr('r', geometry.attnSize / 2)
            .on('click', (event: PointerEvent, p) => {
                handleRepresentationClick(p.node)
            })

        svg
            .selectAll('ffn')
            .data(nodeProps)
            .enter()
            .filter((p) => p.node.item === CellItem.Ffn && p.isActive)
            .append('rect')
            .attr('class', (p) => getNodeStyle(p, 'ffn'))
            .attr('x', (p) => p.pos.x - geometry.ffnSize / 2)
            .attr('y', (p) => p.pos.y - geometry.ffnSize / 2)
            .attr('width', geometry.ffnSize)
            .attr('height', geometry.ffnSize)
            .on('click', (event: PointerEvent, p) => {
                handleRepresentationClick(p.node)
            })

        svg
            .selectAll('token_labels')
            .data(tokenLabels)
            .enter()
            .append('text')
            .attr('x', (label: Label) => label.pos.x)
            .attr('y', (label: Label) => label.pos.y)
            .attr('text-anchor', isAudioTimeline ? 'middle' : 'end')
            .attr('dominant-baseline', isAudioTimeline ? 'hanging' : 'middle')
            .attr('alignment-baseline', isAudioTimeline ? 'hanging' : 'bottom')
            .attr('transform', (label: Label) => (
                isAudioTimeline
                    ? null
                    : 'rotate(40, ' + label.pos.x + ', ' + label.pos.y + ')'
            ))
            .attr('font-size', isAudioTimeline ? 11 : 14)
            .text((label: Label) => label.text)

        if (isAudioTimeline) {
            svg
                .append('line')
                .attr('x1', xScale(0))
                .attr('x2', xScale(Math.max(nTokens, 1)))
                .attr('y1', audioAxisY)
                .attr('y2', audioAxisY)
                .attr('stroke', '#b8c2cc')
                .attr('stroke-width', 1)

            svg
                .selectAll('audio_axis_ticks')
                .data(tokenLabels)
                .enter()
                .append('line')
                .attr('x1', (label: Label) => label.pos.x)
                .attr('x2', (label: Label) => label.pos.x)
                .attr('y1', audioAxisY)
                .attr('y2', audioAxisY + 6)
                .attr('stroke', '#b8c2cc')
                .attr('stroke-width', 1)
        }

        svg
            .selectAll('layer_labels')
            .data(layerLabels)
            .enter()
            .append('text')
            .attr('x', (label: Label) => label.pos.x)
            .attr('y', (label: Label) => label.pos.y)
            .attr('text-anchor', 'middle')
            .attr('alignment-baseline', 'middle')
            .text((label: Label) => label.text)

        svg
            .selectAll('token_selectors')
            .data(tokenSelectors)
            .enter()
            .append('polygon')
            .attr('class', ([i,]) => (
                curStartToken === i
                    ? 'selectable-item selection'
                    : 'selectable-item token-selector'
            ))
            .attr('points', ([, p]) => tokenPointerPolygon(p, geometry.tokenSelectorSize))
            .attr('r', geometry.tokenSelectorSize / 2)
            .on('click', (event: PointerEvent, [i,]) => {
                handleTokenClick(i)
            })
    }, [
        cells,
        currentEdges,
        geometry.attnSize,
        geometry.cellW,
        geometry.ffnSize,
        geometry.leftPad,
        geometry.tokenSelectorSize,
        nodeProps,
        tokenLabels,
        layerLabels,
        tokenSelectors,
        curStartToken,
        curSelection,
        colorScale,
        ffnEdgeColorScale,
        edgeWidthScale,
        audioAxisY,
        isAudioTimeline,
        nLayers,
        nTokens,
        xScale,
        yScale
    ])

    return (
        <div
            ref={containerRef}
            className="graph-scroll-container"
        >
            <svg ref={svgRef} className="graph-svg" width={totalW} height={totalH}></svg>
        </div>
    )
}

export default withStreamlitConnection(ContributionGraph)
