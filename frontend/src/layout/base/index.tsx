import iconNewchat from '@/assets/layout/newchat.svg'
import { useResizablePanel } from '@/hooks/use-resizable-panel'
import iconRepository from '@/assets/layout/repository.svg'
import { deviceState } from '@/store/device'
import classNames from 'classnames'
import { CSSProperties, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useSnapshot } from 'valtio'
import { Background } from './background'
import { Footer } from './footer'
import './index.scss'
import { Nav } from './nav'

const TITLE = import.meta.env.VITE_TITLE

export function BaseLayout({ children }: { children?: React.ReactNode }) {
  const navigate = useNavigate()
  const device = useSnapshot(deviceState)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const {
    size: sidebarWidth,
    isResizing: isSidebarResizing,
    handlePointerDown: handleSidebarPointerDown,
    handleKeyDown: handleSidebarKeyDown,
  } = useResizablePanel({
    defaultSize: 206,
    minSize: 160,
    maxSize: 420,
    resizeEdge: 'right',
    storageKey: 'rag-system-left-sidebar-width',
  })
  const layoutStyle = {
    '--base-layout-sidebar-width': `${sidebarWidth}px`,
  } as CSSProperties

  return (
    <div
      className={classNames('base-layout', {
        'base-layout--sidebar-collapsed': sidebarCollapsed,
        'base-layout--sidebar-resizing': isSidebarResizing,
      })}
      style={layoutStyle}
    >
      <div className="base-layout__sidebar">
        <button
          type="button"
          className="base-layout__sidebar-toggle"
          aria-label={sidebarCollapsed ? '展开左侧栏' : '折叠左侧栏'}
          aria-expanded={!sidebarCollapsed}
          title={sidebarCollapsed ? '展开左侧栏' : '折叠左侧栏'}
          onClick={() => setSidebarCollapsed((collapsed) => !collapsed)}
        >
          {sidebarCollapsed ? '›' : '‹'}
        </button>

        <div
          className="base-layout__sidebar-resizer"
          role="separator"
          aria-label="调整左侧栏宽度"
          aria-orientation="vertical"
          aria-valuemin={160}
          aria-valuemax={420}
          aria-valuenow={sidebarWidth}
          tabIndex={sidebarCollapsed ? -1 : 0}
          title="拖动以调整左侧栏宽度"
          onPointerDown={handleSidebarPointerDown}
          onKeyDown={handleSidebarKeyDown}
        />

        <div className="base-layout__logo">
          <span className="title">{TITLE}</span>
        </div>

        <div className="base-layout__sidebar-main scrollbar-style">
          <div className="base-layout__sidebar-main-content">
            <div
              className="base-layout__nav-header"
              onClick={() => (device.chatting ? null : navigate('/'))}
            >
              <img className="base-layout__nav-header-icon" src={iconNewchat} />
              <span className="base-layout__nav-header-title">新对话</span>
            </div>

            <Nav />

            <div
              className="base-layout__nav-header"
              onClick={() => (device.chatting ? null : navigate('/repository'))}
            >
              <img
                className="base-layout__nav-header-icon"
                src={iconRepository}
              />
              <span className="base-layout__nav-header-title">知识库</span>
            </div>
          </div>

          <Footer />
        </div>
      </div>

      <div className="base-layout__content">{children}</div>

      <Background />
    </div>
  )
}
