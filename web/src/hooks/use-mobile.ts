import * as React from "react"

const MOBILE_BREAKPOINT = 768

function subscribeToMediaQuery(query: string, onChange: () => void) {
  const mediaQueryList = window.matchMedia(query)
  const listener = () => {
    onChange()
  }

  mediaQueryList.addEventListener("change", listener)

  return () => {
    mediaQueryList.removeEventListener("change", listener)
  }
}

function getMediaQueryMatch(query: string) {
  if (typeof window === "undefined") {
    return false
  }

  return window.matchMedia(query).matches
}

export function useMediaQuery(query: string) {
  const [matches, setMatches] = React.useState(() => getMediaQueryMatch(query))

  React.useEffect(() => {
    const updateMatch = () => {
      setMatches(getMediaQueryMatch(query))
    }

    updateMatch()
    return subscribeToMediaQuery(query, updateMatch)
  }, [query])

  return matches
}

export function useIsMobile() {
  return useMediaQuery(`(max-width: ${MOBILE_BREAKPOINT - 1}px)`)
}
