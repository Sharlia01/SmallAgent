declare namespace API {
  interface Session {
    created_at: string
    session_id: string
    session_name: string
    updated_at: string
    user_id: number
  }

  interface ChatItem {
    id: number
    role: import('@/configs').ChatRole
    type: import('@/configs').ChatType
    loading?: boolean
    error?: string
    content?: string
    think?: string

    documents?: Document[]
    reference?: Reference[]
    recommended_questions?: string[]
  }

  interface Document {
    document_id: string
    document_name: string
    content_with_weight: string
  }

  interface Reference {
    id: string
    document_id: string
    document_name: string
    content_with_weight: string
    positions: number[][]
    source_type?: 'knowledge_base' | 'web'
    url?: string
    web_sources?: {
      source_type: 'web'
      source_id: string
      title: string
      content: string
      url?: string
      metadata: {
        site_name?: string
        icon?: string
        index?: number
      }
    }[]
  }
}
