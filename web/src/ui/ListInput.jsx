import { useEffect, useState } from 'react';
import { Button, Card, Flex, Heading, Text, TextArea } from '@radix-ui/themes';
import { UploadIcon } from '@radix-ui/react-icons';

import { LiveRegion } from './LiveRegion.jsx';

/** One place decides how a count of entries reads, including the singular. */
export const entryLabel = (n) => `${n.toLocaleString()} ${n === 1 ? 'entry' : 'entries'}`;

/** One of the two entry lists. Extracted so both views share the same shell. */
export function ListInput({ title, hint, placeholder, value, onChange, count }) {
  // announce on a pause; one announcement per keystroke is noise to sit through
  const [announced, setAnnounced] = useState(count);
  useEffect(() => {
    const timer = setTimeout(() => setAnnounced(count), 700);
    return () => clearTimeout(timer);
  }, [count]);

  return (
    <Card size="3" variant="surface">
      <Flex direction="column" gap="3">
        <Flex align="baseline" justify="between" gap="3">
          <Heading as="h2" size="3" weight="medium">{title}</Heading>
          <Text size="1" color="gray" className="tabular">
            {count > 0 ? entryLabel(count) : ''}
          </Text>
        </Flex>
        <Text size="2" color="gray">{hint}</Text>
        <TextArea
          size="3"
          rows={8}
          spellCheck={false}
          placeholder={placeholder}
          value={value}
          aria-label={title}
          onChange={(e) => onChange(e.target.value)}
          style={{ fontFamily: 'var(--code-font-family)' }}
        />
        <LiveRegion>{announced > 0 ? `${title}: ${entryLabel(announced)}` : ''}</LiveRegion>
        <Flex align="center" gap="2">
          {/* The input lives inside the label rather than beside it, and is clipped
              rather than display:none. display:none takes it out of the tab order, which
              silently made this control mouse-only: a <label> is not focusable, so there
              was nothing left for a keyboard to reach. */}
          <Button asChild variant="soft" size="1" color="gray" className="file-trigger">
            <label style={{ cursor: 'pointer' }}>
              <UploadIcon aria-hidden /> Load a file
              <input
                type="file"
                className="visually-hidden"
                accept=".txt,.csv,.tsv"
                onChange={async (e) => {
                  const file = e.target.files?.[0];
                  if (file) onChange(await file.text());
                  // Clearing lets the same file be picked again after an edit.
                  e.target.value = '';
                }}
              />
            </label>
          </Button>
          <Text size="1" color="gray">.txt, .csv or .tsv</Text>
        </Flex>
      </Flex>
    </Card>
  );
}
